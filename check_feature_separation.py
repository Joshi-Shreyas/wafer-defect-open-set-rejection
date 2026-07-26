"""
check_feature_separation.py

Tests whether the raw pretrained FEATURES (before the classification head) separate
known vs. unknown better than the energy score (which comes from post-classification
logits). Uses the raw_scores.npz files already saved by evaluate.py for both regimes.

If Mahalanobis-in-feature-space shows better separation than the energy/logit-based
AUROC we already measured, that supports the hypothesis that Regime B's fine-tuned
classification head is discarding separability that exists in the underlying
MAE-pretrained features.
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.covariance import EmpiricalCovariance


def mahalanobis_auroc(test_features, test_labels, unknown_features, num_classes=8):
    """
    Computes per-class centroids and a shared covariance from the known test features,
    then scores every sample (known and unknown) by its minimum Mahalanobis distance
    to any known-class centroid. Higher distance = more 'unknown-like'.
    """
    centroids = []
    for c in range(num_classes):
        class_feats = test_features[test_labels == c]
        centroids.append(class_feats.mean(axis=0))
    centroids = np.stack(centroids)

    cov_estimator = EmpiricalCovariance().fit(test_features)
    precision = cov_estimator.precision_  # inverse covariance

    def min_mahalanobis(feats):
        dists = []
        for c in range(num_classes):
            diff = feats - centroids[c]
            d = np.sqrt(np.sum(diff @ precision * diff, axis=1))
            dists.append(d)
        return np.min(np.stack(dists, axis=0), axis=0)

    known_dist = min_mahalanobis(test_features)
    unknown_dist = min_mahalanobis(unknown_features)

    binary_labels = np.concatenate([np.zeros(len(known_dist)), np.ones(len(unknown_dist))])
    binary_scores = np.concatenate([known_dist, unknown_dist])
    auroc = roc_auc_score(binary_labels, binary_scores)

    return auroc, known_dist, unknown_dist


for regime in ['supervised', 'finetuned']:
    path = f'/scratch/joshi.shreyas/wafer_results/eval_{regime}/raw_scores.npz'
    data = np.load(path)

    test_features = data['test_features']
    test_labels = data['test_labels']
    unknown_features = data['unknown_features']

    auroc, known_dist, unknown_dist = mahalanobis_auroc(test_features, test_labels, unknown_features)

    print(f"\n=== {regime.upper()} ===")
    print(f"Known Mahalanobis dist   - mean: {known_dist.mean():.3f}, std: {known_dist.std():.3f}")
    print(f"Unknown Mahalanobis dist - mean: {unknown_dist.mean():.3f}, std: {unknown_dist.std():.3f}")
    print(f"Mahalanobis (feature-space) AUROC: {auroc:.4f}")
    print(f"(Compare to energy/logit-space AUROC from evaluate.py: "
          f"{'0.7885' if regime == 'supervised' else '0.5019'})")

    # Cost-sensitive threshold sweep using Mahalanobis distance instead of energy score
    for fn_cost in [20.0, 100.0]:
        all_thresholds = np.linspace(known_dist.min(), unknown_dist.max(), 300)
        best_cost = float('inf')
        best_threshold = None

        binary_labels = np.concatenate([np.zeros(len(known_dist)), np.ones(len(unknown_dist))])
        binary_scores = np.concatenate([known_dist, unknown_dist])

        for thresh in all_thresholds:
            rejected = binary_scores > thresh
            fp = ((binary_labels == 0) & rejected).sum()
            fn = ((binary_labels == 1) & ~rejected).sum()
            total_cost = fn_cost * fn + 1.0 * fp
            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = thresh

        rejected_at_best = binary_scores > best_threshold
        fp_best = int(((binary_labels == 0) & rejected_at_best).sum())
        fn_best = int(((binary_labels == 1) & ~rejected_at_best).sum())
        caught = len(unknown_dist) - fn_best

        print(f"\n  [Mahalanobis cost sweep, fn_cost={fn_cost}]")
        print(f"  Best threshold: {best_threshold:.3f} | Caught: {caught}/{len(unknown_dist)} novel defects | "
              f"FPs: {fp_best} | Total cost: {best_cost:.1f}")
