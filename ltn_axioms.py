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


def _axiom(p_high, phys_truth):
    """forall x: HighLoad(x) -> Phys(x), grounded on the batch."""
    val = torch.stack([p_high, phys_truth], dim=1)          # (batch, 2)
    x = ltn.Variable("x", val, add_batch_dim=False)
    HighLoad = ltn.Predicate(func=lambda v: v[:, 0])        # P(high|x) from the network
    Phys = ltn.Predicate(func=lambda v: v[:, 1])            # fuzzy physiological truth
    return _Forall(x, _Implies(HighLoad(x), Phys(x)))


def ltn_physio_loss(p_high, hrv, baseline, idxs, k: float = 2.0):
    """
    Returns 1 - satisfaction of the physiological axioms (a scalar to add to the loss).
    p_high : (batch,) network probability of the HIGH-load class.
    hrv, baseline : (batch, F) STANDARDIZED HRV of the window and the subject baseline.
    idxs : output of build_axiom_indices.
    """
    axioms = []
    if idxs["mean"] is not None:
        d = hrv[:, idxs["mean"]] - baseline[:, idxs["mean"]]
        axioms.append(_axiom(p_high, torch.sigmoid(-k * d)))   # HR up  <=> MeanNN down
    if idxs["rmssd"] is not None:
        d = hrv[:, idxs["rmssd"]] - baseline[:, idxs["rmssd"]]
        axioms.append(_axiom(p_high, torch.sigmoid(-k * d)))   # RMSSD down
    if idxs["lfhf"] is not None:
        d = hrv[:, idxs["lfhf"]] - baseline[:, idxs["lfhf"]]
        axioms.append(_axiom(p_high, torch.sigmoid(k * d)))    # LF/HF up
    if not axioms:
        return torch.zeros((), device=p_high.device)
    return 1.0 - _SatAgg(*axioms)


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
    print("ltn loss:", float(loss), "| shape:", tuple(loss.shape))
    assert loss.ndim == 0 and 0.0 <= float(loss) <= 1.0, "loss must be a scalar in [0,1]"
    loss.backward()
    print("grad flows to logits:", logits.grad is not None and torch.isfinite(logits.grad).all().item())
    print("LTN axiom self-test passed.")
