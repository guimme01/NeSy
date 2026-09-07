"""
Neuro-Symbolic upgrade via Logic Tensor Networks (LTNtorch).

Physiological domain knowledge is expressed as first-order fuzzy-logic axioms that
constrain the model's HIGH-load predictions, and enforced through a differentiable
satisfaction loss added to the cross-entropy:

    loss = CrossEntropy  +  lambda * (1 - SatAgg(axioms))

Axioms (each a soft universally-quantified implication, Reichenbach implication):
    forall x:  HighLoad(x) -> HR_up(x)       # cognitive load raises heart rate (MeanNN drops)
    forall x:  HighLoad(x) -> RMSSD_down(x)  # parasympathetic withdrawal (RMSSD drops)
    forall x:  HighLoad(x) -> LFHF_up(x)     # sympathetic shift (LF/HF rises)

HighLoad(x) is grounded in the network's own P(high | x), so gradients push the model
to make predictions that are consistent with physiology. The physiological predicates
are fuzzy functions of the STANDARDIZED HRV shift vs. the subject's own baseline.
"""

import numpy as np
import torch
import ltn

# LTN operators (Reichenbach implication I(a,b)=1-a+a*b; p-mean-error universal aggregator).
_Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
_Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
_SatAgg = ltn.fuzzy_ops.SatAgg()


def build_axiom_indices(hrv_feature_names):
    """Locate the HRV columns used by the axioms (None if a feature is absent)."""
    def idx(name):
        return hrv_feature_names.index(name) if name in hrv_feature_names else None
    return {"mean": idx("HRV_MeanNN"), "rmssd": idx("HRV_RMSSD"), "lfhf": idx("HRV_LFHF")}


def _implies_axiom(antecedent, consequent):
    """forall x: A(x) -> B(x), grounded on the batch (Reichenbach implication).
    Returns the axiom's satisfaction as a scalar tensor (LTNObject.value)."""
    val = torch.stack([antecedent, consequent], dim=1)      # (batch, 2)
    x = ltn.Variable("x", val, add_batch_dim=False)
    A = ltn.Predicate(func=lambda v: v[:, 0])
    B = ltn.Predicate(func=lambda v: v[:, 1])
    return _Forall(x, _Implies(A(x), B(x))).value


def _predicate_truths(hrv, baseline, idxs, k):
    """Fuzzy truth of each physiological predicate on the batch (dict name -> tensor)."""
    truths = {}
    if idxs["mean"] is not None:
        d = hrv[:, idxs["mean"]] - baseline[:, idxs["mean"]]
        truths["hr_up"] = torch.sigmoid(-k * d)              # HR up  <=> MeanNN down
    if idxs["rmssd"] is not None:
        d = hrv[:, idxs["rmssd"]] - baseline[:, idxs["rmssd"]]
        truths["rmssd_down"] = torch.sigmoid(-k * d)         # RMSSD down
    if idxs["lfhf"] is not None:
        d = hrv[:, idxs["lfhf"]] - baseline[:, idxs["lfhf"]]
        truths["lfhf_up"] = torch.sigmoid(k * d)             # LF/HF up
    return truths


def axiom_weights(hrv_train, baseline_train, y_train, idxs, k: float = 2.0):
    """
    Empirical validation of the knowledge base on the TRAINING fold only (leakage-free).

    Each axiom's predicate is scored as a standalone classifier of the training labels;
    its weight is w = clip(2*(AUROC - 0.5), 0, 1): an axiom whose predicate carries no
    signal on this paradigm (AUROC <= 0.5) is silenced, a perfectly aligned one gets
    weight 1. This lets the fuzzy KB down-weight clinically canonical axioms that the
    data refute (e.g. RMSSD under this manipulation) instead of injecting their noise.
    """
    from sklearn.metrics import roc_auc_score
    hrv_t = torch.as_tensor(np.asarray(hrv_train), dtype=torch.float32)
    base_t = torch.as_tensor(np.asarray(baseline_train), dtype=torch.float32)
    truths = _predicate_truths(hrv_t, base_t, idxs, k)
    y = np.asarray(y_train)
    weights = {}
    for name, t in truths.items():
        if len(np.unique(y)) < 2:
            weights[name] = 1.0
            continue
        auc = roc_auc_score(y, t.numpy())
        weights[name] = float(np.clip(2.0 * (auc - 0.5), 0.0, 1.0))
    return weights


def ltn_physio_loss(p_high, hrv, baseline, idxs, k: float = 2.0,
                    bidirectional: bool = True, weights=None):
    """
    Returns 1 - satisfaction of the physiological axioms (a scalar to add to the loss).
    p_high : (batch,) network probability of the HIGH-load class.
    hrv, baseline : (batch, F) STANDARDIZED HRV of the window and the subject baseline.
    idxs : output of build_axiom_indices.
    weights : optional dict from axiom_weights() (TRAIN-fold empirical validation);
              None gives every axiom weight 1 (the original untempered KB).

    bidirectional=False reproduces the original one-directional knowledge base
    (HighLoad -> marker), which can only LOWER p_high: with Reichenbach implication
    I(a,b)=1-a(1-b), dI/da = -(1-b) <= 0, so its gradient is a one-way brake on the
    positive class (degenerate optimum: always predict low -> satisfaction 1).

    bidirectional=True (default) closes the loop with the converse axiom
        forall x: AND_i marker_i(x)  ->  HighLoad(x)
    over the validated markers (weighted product t-norm: t_i^{w_i}, so a silenced
    axiom drops out of the conjunction). Concordant physiological evidence then
    PUSHES p_high up -- per-window information the subject-constant CE labels lack.

    Aggregation is a weight-normalized mean of the per-axiom satisfactions, so a
    data-refuted axiom (w=0) contributes neither satisfaction nor gradient.
    """
    truths = _predicate_truths(hrv, baseline, idxs, k)
    if not truths:
        return torch.zeros((), device=p_high.device)
    if weights is None:
        weights = {name: 1.0 for name in truths}

    sats, ws = [], []
    conj, w_conj = None, 0.0
    for name, t in truths.items():
        w = float(weights.get(name, 1.0))
        if w <= 0.0:
            continue
        # Forward axiom: forall x: HighLoad(x) -> marker(x).
        sats.append(_implies_axiom(p_high, t))
        ws.append(w)
        # Weighted product t-norm for the converse antecedent (t^w -> 1 as w -> 0).
        term = t ** w
        conj = term if conj is None else conj * term
        w_conj = max(w_conj, w)

    if not sats:
        return torch.zeros((), device=p_high.device)

    if bidirectional and conj is not None:
        sats.append(_implies_axiom(conj, p_high))
        ws.append(w_conj)

    total_w = sum(ws)
    sat = sum(w * s for w, s in zip(ws, sats)) / total_w
    return 1.0 - sat


if __name__ == "__main__":
    torch.manual_seed(0)
    B, F = 8, 10
    idxs = {"mean": 0, "rmssd": 1, "lfhf": 2}

    # p_high requires grad (stands in for the network output).
    logits = torch.randn(B, 2, requires_grad=True)
    p_high = torch.softmax(logits, dim=1)[:, 1]
    hrv = torch.randn(B, F)
    baseline = torch.randn(B, F)

    loss = ltn_physio_loss(p_high, hrv, baseline, idxs)
    print("ltn loss (bidirectional):", float(loss), "| shape:", tuple(loss.shape))
    assert loss.ndim == 0 and 0.0 <= float(loss) <= 1.0, "loss must be a scalar in [0,1]"
    loss.backward()
    print("grad flows to logits:", logits.grad is not None and torch.isfinite(logits.grad).all().item())

    # --- Directionality checks -------------------------------------------------
    # Window with STRONG concordant evidence (HR up, RMSSD down, LF/HF up) but a
    # LOW predicted p_high: the bidirectional KB must push p_high UP (negative
    # gradient of the loss wrt p_high), while the one-directional KB cannot.
    def grad_on_phigh(bidirectional):
        ph = torch.tensor([0.05], requires_grad=True)
        h = torch.zeros(1, F); b = torch.zeros(1, F)
        h[0, 0] = -3.0   # MeanNN well below baseline  -> HR up
        h[0, 1] = -3.0   # RMSSD well below baseline   -> RMSSD down
        h[0, 2] = +3.0   # LF/HF well above baseline   -> LF/HF up
        loss = ltn_physio_loss(ph, h, b, idxs, bidirectional=bidirectional)
        loss.backward()
        return float(ph.grad)

    g_bi, g_uni = grad_on_phigh(True), grad_on_phigh(False)
    print(f"d(loss)/d(p_high) on an active window: bidirectional={g_bi:+.4f}  "
          f"one-directional={g_uni:+.4f}")
    assert g_bi < 0, "bidirectional KB must push p_high UP on concordant evidence"
    assert g_uni >= -1e-8, "one-directional KB can never push p_high up"

    # --- Weight validation checks ----------------------------------------------
    # Synthetic training fold where marker 0 is aligned with the label, marker 1 is
    # pure noise, marker 2 anti-aligned: weights must be ~1 / ~0 / 0.
    rng = np.random.RandomState(0)
    n = 4000
    y_syn = rng.randint(0, 2, n)
    h = np.zeros((n, F), dtype=np.float32)
    b = np.zeros((n, F), dtype=np.float32)
    h[:, 0] = -2.0 * y_syn + rng.randn(n) * 0.3          # MeanNN down when y=1 -> HR_up aligned
    h[:, 1] = rng.randn(n)                                # RMSSD column: noise
    h[:, 2] = -2.0 * y_syn + rng.randn(n) * 0.3          # LF/HF DOWN when y=1 -> anti-aligned
    w = axiom_weights(h, b, y_syn, idxs)
    print("axiom weights (aligned/noise/anti):",
          {k_: round(v, 3) for k_, v in w.items()})
    assert w["hr_up"] > 0.9 and w["rmssd_down"] < 0.15 and w["lfhf_up"] == 0.0

    # A zero-weight axiom must contribute no gradient: loss must be invariant to the
    # silenced marker's values.
    ph = torch.tensor([0.3, 0.7], requires_grad=True)
    h2 = torch.zeros(2, F); h2[:, 0] = -1.0
    h2b = h2.clone(); h2b[:, 2] = 99.0                    # perturb only the silenced marker
    l1 = ltn_physio_loss(ph, h2, torch.zeros(2, F), idxs, weights=w)
    l2 = ltn_physio_loss(ph, h2b, torch.zeros(2, F), idxs, weights=w)
    assert torch.allclose(l1, l2), "silenced axiom must not affect the loss"
    print("LTN axiom self-test passed (upward gradient + empirical KB validation).")
