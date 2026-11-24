"""Public binary metrics, shared by local validation and the private scorer."""
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


def binary_metrics(labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities) or not len(labels):
        raise ValueError('Expected equally sized, nonempty, one-dimensional inputs')
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Labels must be 0 (normal) or 1 (tumor)')
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Tumor probabilities must be finite and between 0 and 1')
    if set(labels.tolist()) != {0, 1}:
        raise ValueError('Both classes are required for balanced accuracy and AUROC')
    predicted = (probabilities >= 0.5).astype(int)
    return {'balanced_accuracy': float(balanced_accuracy_score(labels, predicted)),
            'auroc': float(roc_auc_score(labels, probabilities)),
            'accuracy': float(accuracy_score(labels, predicted)),
            'n_samples': int(len(labels))}
