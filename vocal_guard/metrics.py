import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and threshold.

    In ASVspoof:
        - bonafide is target (positive class, higher score = more likely bonafide)
        - spoof is non-target (negative class, lower score)

    Returns:
        eer: Equal Error Rate as a percentage (e.g., 5.12 for 5.12%) or fraction in [0, 1]
             Here returning fraction in [0, 1] and percentage is eer * 100
        threshold: score threshold where FAR == FRR
    """
    y_true = np.concatenate([np.ones_like(bonafide_scores), np.zeros_like(spoof_scores)])
    scores = np.concatenate([bonafide_scores, spoof_scores])

    # fpr: spoof accepted as bonafide (False Acceptance Rate - FAR)
    # tpr: bonafide accepted as bonafide (1 - False Rejection Rate = 1 - FRR)
    # thresholds sorted descending
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1.0 - tpr  # False Rejection Rate (FRR)

    # Find threshold index where |fpr - fnr| is minimized
    abs_diff = np.abs(fpr - fnr)
    idx = np.argmin(abs_diff)

    # Exact EER estimation by linear interpolation
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]

    return float(eer), float(threshold)
