"""Allocation diagnostics (protocol section 7, equations 17-20).

I is the binary ROI (b = 1[m >= 0.1]) and G1 the full-step (HIGH) group.

    P = |G1 & I| / |G1|        R = |G1 & I| / |I|        R_norm = R / min(1, |G1| / |I|)   (eq. 17)
    lift = P / (|I| / N)       A_ROI = share of ROI tokens whose group differs from C3   (eq. 18)
    S_ROI = sum_n |A_n & I| / sum_n |A_n|                                                (eq. 19)
    no-op  <=>  no token's group differs from C3                                         (eq. 20)

The set-based quantities are undefined for an empty ROI and reported as None
(not applicable). Integer counts are returned alongside the ratios.
"""

import torch

from density.hsa import HIGH


def allocation_diagnostics(
    labels: torch.Tensor,
    roi: torch.Tensor,
    active: torch.Tensor | None = None,
    reference_labels: torch.Tensor | None = None,
) -> dict:
    """Diagnostics for one allocation.

    labels: group label per token; roi: binary ROI per token; active: bool
    [iterations, tokens] active-token matrix, for S_ROI; reference_labels: C3's
    labels, for A_ROI and the no-op test.
    """
    labels, roi = labels.cpu(), roi.cpu().bool()
    full_step = labels == HIGH
    num_tokens = labels.numel()
    roi_tokens = int(roi.sum())
    full_tokens = int(full_step.sum())
    full_roi = int((full_step & roi).sum())

    out = {"tokens": num_tokens, "roi_tokens": roi_tokens, "full_step_tokens": full_tokens, "full_step_roi_tokens": full_roi}
    empty = roi_tokens == 0
    out["precision"] = None if empty or full_tokens == 0 else full_roi / full_tokens
    out["recall"] = None if empty else full_roi / roi_tokens
    out["normalized_recall"] = None if empty else (full_roi / roi_tokens) / min(1.0, full_tokens / roi_tokens)
    out["lift"] = None if out["precision"] is None else out["precision"] / (roi_tokens / num_tokens)

    if active is not None:
        active = active.cpu().bool()
        out["roi_evaluations"] = int((active & roi).sum())
        out["evaluations"] = int(active.sum())
        out["s_roi"] = None if empty else out["roi_evaluations"] / out["evaluations"]

    if reference_labels is not None:
        changed = labels != reference_labels.cpu()
        out["changed_tokens"] = int(changed.sum())
        out["changed_roi_tokens"] = int((changed & roi).sum())
        out["a_roi"] = None if empty else out["changed_roi_tokens"] / roi_tokens
        out["no_op"] = out["changed_tokens"] == 0
    return out
