"""Baseline models for TurnZero: popularity and multinomial logistic regression.

Both baselines output (N, 90) probability arrays so they plug directly into
the eval harness (``compute_metrics``).

Reference: docs/PROJECT_BIBLE.md Section 3 — Mandatory baselines
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy import sparse


# ---------------------------------------------------------------------------
# Baseline 1: Popularity / frequency
# ---------------------------------------------------------------------------

class PopularityBaseline:
    """Global P(action90) from training-set frequencies.

    Also supports core-conditional variant
    ``P(action90 | core_cluster_a, core_cluster_b)`` with Laplace smoothing.
    """

    N_ACTIONS: int = 90

    def __init__(self) -> None:
        self._global_probs: np.ndarray | None = None
        # core-conditional: (cluster_a, cluster_b) -> (90,) prob vector
        self._cond_probs: dict[tuple[str, str], np.ndarray] = {}
        self._laplace_alpha: float = 1.0

    # -- Global fit / predict -----------------------------------------------

    def fit(self, action90_labels: np.ndarray) -> PopularityBaseline:
        """Fit global P(action90) from training labels."""
        labels = np.asarray(action90_labels, dtype=np.int64)
        counts = np.bincount(labels, minlength=self.N_ACTIONS).astype(np.float64)
        self._global_probs = counts / counts.sum()
        return self

    def predict(self, n: int) -> np.ndarray:
        """Broadcast the global prob vector n times → (n, 90)."""
        assert self._global_probs is not None, "Call fit() first"
        return np.tile(self._global_probs, (n, 1))

    # -- Core-conditional fit / predict -------------------------------------

    def fit_conditional(
        self,
        action90_labels: np.ndarray,
        cluster_a: list[str],
        cluster_b: list[str],
        alpha: float = 1.0,
    ) -> PopularityBaseline:
        """Fit P(action90 | core_cluster_a, core_cluster_b) with Laplace smoothing."""
        labels = np.asarray(action90_labels, dtype=np.int64)
        self._laplace_alpha = alpha

        # Count per (cluster_a, cluster_b) pair
        pair_counts: dict[tuple[str, str], np.ndarray] = defaultdict(
            lambda: np.zeros(self.N_ACTIONS, dtype=np.float64)
        )
        for lab, ca, cb in zip(labels, cluster_a, cluster_b):
            pair_counts[(ca, cb)][lab] += 1.0

        # Normalize with Laplace smoothing
        self._cond_probs = {}
        for key, counts in pair_counts.items():
            smoothed = counts + alpha
            self._cond_probs[key] = smoothed / smoothed.sum()

        # Also fit global as fallback
        if self._global_probs is None:
            self.fit(labels)

        return self

    def predict_conditional(
        self, cluster_a: str, cluster_b: str
    ) -> np.ndarray:
        """Return (90,) prob vector for a specific cluster pair.

        Falls back to global probs if the pair is unseen.
        """
        if (cluster_a, cluster_b) in self._cond_probs:
            return self._cond_probs[(cluster_a, cluster_b)]
        assert self._global_probs is not None, "Call fit() or fit_conditional() first"
        return self._global_probs.copy()

    def predict_conditional_batch(
        self, cluster_a_list: list[str], cluster_b_list: list[str]
    ) -> np.ndarray:
        """Return (N, 90) prob array for a batch of cluster pairs."""
        return np.array(
            [
                self.predict_conditional(ca, cb)
                for ca, cb in zip(cluster_a_list, cluster_b_list)
            ]
        )


# ---------------------------------------------------------------------------
# Baseline 2: Multinomial logistic regression on one-hot OTS features
# ---------------------------------------------------------------------------

class LogisticBaseline:
    """Multinomial logistic regression on one-hot OTS features.

    Features are bag-of-features per team side (NOT per slot):
    species presence, item presence, ability presence, tera presence,
    move presence — concatenated for side A and side B.
    """

    N_ACTIONS: int = 90

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self._model = None
        self._feature_names: list[str] = []
        self._feat2idx: dict[str, int] = {}

    # -- Feature extraction -------------------------------------------------

    @staticmethod
    def _extract_team_features(
        team_dict: dict[str, Any], side: str
    ) -> list[str]:
        """Extract bag-of-features keys from a TeamSheet dict.

        Returns feature strings like "a:species=Incineroar", "b:move=Fake Out".
        """
        features: list[str] = []
        for mon in team_dict["pokemon"]:
            features.append(f"{side}:species={mon['species']}")
            features.append(f"{side}:item={mon['item']}")
            features.append(f"{side}:ability={mon['ability']}")
            features.append(f"{side}:tera={mon['tera_type']}")
            for m in mon["moves"]:
                if m != "UNK":
                    features.append(f"{side}:move={m}")
        return features

    def build_feature_index(self, examples: list[dict[str, Any]]) -> None:
        """Build the feature-name-to-column-index mapping from training data."""
        feat_set: set[str] = set()
        for ex in examples:
            feat_set.update(self._extract_team_features(ex["team_a"], "a"))
            feat_set.update(self._extract_team_features(ex["team_b"], "b"))
        self._feature_names = sorted(feat_set)
        self._feat2idx = {f: i for i, f in enumerate(self._feature_names)}

    def featurize(self, examples: list[dict[str, Any]]) -> sparse.csr_matrix:
        """Convert examples to a sparse feature matrix (N, n_features).

        Must call ``build_feature_index`` first (on training data).
        """
        n = len(examples)
        n_feats = len(self._feature_names)
        rows, cols, data = [], [], []

        for i, ex in enumerate(examples):
            feats_a = self._extract_team_features(ex["team_a"], "a")
            feats_b = self._extract_team_features(ex["team_b"], "b")
            # Use set to avoid duplicates from repeated moves across mons
            seen: set[int] = set()
            for f in feats_a + feats_b:
                idx = self._feat2idx.get(f)
                if idx is not None and idx not in seen:
                    rows.append(i)
                    cols.append(idx)
                    data.append(1.0)
                    seen.add(idx)

        return sparse.csr_matrix((data, (rows, cols)), shape=(n, n_feats))

    # -- Fit / predict ------------------------------------------------------

    def fit(
        self, examples: list[dict[str, Any]], action90_labels: np.ndarray
    ) -> LogisticBaseline:
        """Fit multinomial logistic regression on training examples.

        Automatically builds feature index and featurizes.
        """
        from sklearn.linear_model import LogisticRegression

        labels = np.asarray(action90_labels, dtype=np.int64)
        self.build_feature_index(examples)
        X = self.featurize(examples)

        print(f"  LogisticBaseline: {X.shape[0]} examples, {X.shape[1]} features")
        self._model = LogisticRegression(
            max_iter=self.max_iter,
            C=self.C,
            solver=self.solver,
            verbose=1,
        )
        self._model.fit(X, labels)
        return self

    def predict_proba(self, examples: list[dict[str, Any]]) -> np.ndarray:
        """Return (N, 90) probability array for test examples."""
        assert self._model is not None, "Call fit() first"
        X = self.featurize(examples)
        return self._model.predict_proba(X)
