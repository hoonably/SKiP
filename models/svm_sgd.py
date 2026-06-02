"""
Gradient Descent based SVM models with ANN (FAISS) weight computation.

Replaces the QP (cvxpy/OSQP) solver in svm_models.py with Mini-batch SGD
(Pegasos-style) on the primal objective. Replaces exact KNN with FAISS-based
Approximate Nearest Neighbors for scalability to large datasets.

Primal objective (per-sample weighted):
    min_{w,b}  1/2 * lambda * ||w||^2 + (1/n) * sum_i s_i * hinge(y_i*(w^T x_i + b))

where s_i (sample weight) differs per model:
    NaiveSVM_SGD  : s_i = 1  (uniform)
    ProbSVM_SGD   : s_i = p_i  (Mahalanobis-capped typicality weight; ≈1 for clean samples)
    KNNSVM_GD    : s_i = w_i  (ANN label-agreement weight)
    SKiP_SGD      : s_i = w_i * p_i  (combined; preserves clean acc, robust to both noise types)

For RBF kernel, uses budget-constrained Kernelized Pegasos:
    f(x) = sum_{j in S} coeff_j * y_j * K(x_j, x) + b
where S is a bounded set of retained support vectors.
"""

import numpy as np
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn(
        "faiss not found. Falling back to exact sklearn NearestNeighbors. "
        "Install with: pip install faiss-gpu  (or faiss-cpu for CPU-only)",
        ImportWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_faiss_index(X_f32: np.ndarray, k: int):
    """
    Build a FAISS HNSW index on float32 data.

    For n < 10_000 : exact IndexFlatL2 (HNSW overhead not worth it at small scale).
    For n >= 10_000 : IndexHNSWFlat — graph-based ANN with higher recall than IVFFlat.

    HNSW parameters:
        M             = 32   (edges per node; higher → better recall, more memory)
        efConstruction= 200  (build quality; higher → better graph, slower build)
        efSearch      = 64   (query quality; higher → better recall, slower query)

    Note: HNSW does not support GPU transfer; no training step needed.

    Returns the populated index.
    """
    n, d = X_f32.shape
    M              = 32
    efConstruction = 200
    efSearch       = 64
    index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch       = efSearch
    index.add(X_f32)
    return index


def _ann_kneighbors(index, X_f32: np.ndarray, k: int):
    """
    Query the FAISS index for (k+1) neighbors and strip the self-match.
    Returns (distances_l2, indices) each shaped (n, k).
    """
    distances_sq, indices = index.search(X_f32, k + 1)
    # The first neighbor is almost always self (distance ≈ 0).
    return np.sqrt(np.maximum(distances_sq[:, 1:], 0.0)), indices[:, 1:]


def _sklearn_kneighbors(X: np.ndarray, k: int):
    """Exact KNN fallback when FAISS is unavailable."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    return distances[:, 1:], indices[:, 1:]


# ---------------------------------------------------------------------------
# NaiveSVM_SGD  (base class)
# ---------------------------------------------------------------------------

class NaiveSVM_SGD(BaseEstimator, ClassifierMixin):
    """
    Linear/RBF SVM trained with mini-batch SGD (Pegasos) on the primal.

    Linear kernel  → explicit weight vector w ∈ R^d, updated each step.
    RBF kernel     → budget-constrained Kernelized Pegasos:
                     f(x) = Σ coeff_j * y_j * K(x_j, x) + b

    Parameters
    ----------
    C : float
        Regularisation trade-off (dual upper-bound on alpha; primal
        inverse-regularisation strength lambda = 1/C).
    fit_intercept : bool
        Whether to learn a bias term b.
    kernel : {'linear', 'rbf'}
        Kernel type.
    gamma : float or {'scale', 'auto'}
        RBF bandwidth.  'scale' → 1/(n_features * X.var()).
    lr : float
        Base learning rate.  Used directly for 'constant'; as initial cap
        for 'pegasos'.
    lr_schedule : {'pegasos', 'constant', 'sqrt_decay'}
        'pegasos'    → eta_t = min(lr, 1/(lambda*t))   [recommended]
        'constant'   → eta_t = lr
        'sqrt_decay' → eta_t = lr / sqrt(t)
    epochs : int
        Number of full passes over the training data.
    batch_size : int
        Mini-batch size.
    budget : int or None
        Maximum number of retained support vectors for Kernelized Pegasos.
        None means unlimited (may become slow for many iterations).
    verbose : bool
        Print per-epoch loss.
    random_state : int
        Seed for reproducibility.
    class_weight : {'balanced', None}
        'balanced'  → multiply sample weights by n / (2 * class_count).
                      Recommended when used inside OneVsRestSVM, because the
                      primal GD lacks the dual constraint Σαᵢyᵢ=0 that the
                      QP solver uses to handle class imbalance automatically.
        None        → uniform weights (same as QP when classes are balanced).
    """

    def __init__(
        self,
        C=1.0,
        fit_intercept=True,
        kernel='linear',
        gamma='scale',
        lr=0.01,
        lr_schedule='pegasos',
        epochs=100,
        batch_size=32,
        budget=300,
        verbose=False,
        random_state=42,
        class_weight='balanced',
    ):
        self.C = C
        self.fit_intercept = fit_intercept
        self.kernel = kernel
        self.gamma = gamma
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.epochs = epochs
        self.batch_size = batch_size
        self.budget = budget
        self.verbose = verbose
        self.random_state = random_state
        self.class_weight = class_weight

    # ------------------------------------------------------------------
    # Kernel helpers
    # ------------------------------------------------------------------

    def _resolve_gamma(self, X):
        if self.gamma == 'scale':
            return 1.0 / (X.shape[1] * float(X.var()))
        if self.gamma == 'auto':
            return 1.0 / X.shape[1]
        return float(self.gamma)

    def _rbf(self, X1, X2):
        """Compute RBF kernel matrix K(X1, X2), shape (n1, n2)."""
        X1_sq = np.einsum('ij,ij->i', X1, X1)[:, None]
        X2_sq = np.einsum('ij,ij->i', X2, X2)[None, :]
        dist2 = X1_sq + X2_sq - 2.0 * X1 @ X2.T
        return np.exp(-self.gamma_ * np.maximum(dist2, 0.0))

    # ------------------------------------------------------------------
    # Learning rate helper
    # ------------------------------------------------------------------

    def _eta(self, step, lam):
        if self.lr_schedule == 'pegasos':
            return min(self.lr, 1.0 / (lam * step))
        if self.lr_schedule == 'sqrt_decay':
            return self.lr / np.sqrt(step)
        return self.lr   # 'constant'

    # ------------------------------------------------------------------
    # Linear SGD
    # ------------------------------------------------------------------

    def _fit_linear_sgd(self, X, y, sample_weights):
        """
        Mini-batch SGD for weighted linear SVM.

        Objective per step:
            L ≈ lambda/2 * ||w||^2 + (1/|B|) * sum_{i in B} s_i * hinge_i

        Update:
            w ← (1 - eta*lambda) * w + (eta/m) * sum_{i∈B:hinge} s_i * y_i * x_i
            b ← b + (eta/m) * sum_{i∈B:hinge} s_i * y_i

        Note: we divide by batch size m (not by |hinge|) so the gradient is an
        unbiased estimator of the full-dataset gradient regardless of how many
        samples violate the margin.
        """
        n, d = X.shape
        lam = 1.0 / self.C
        rng = np.random.default_rng(self.random_state)

        w = np.zeros(d)
        b = 0.0
        losses = []
        step = 0

        log_every = max(1, self.epochs // 10)

        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            epoch_hinge = 0.0

            for start in range(0, n, self.batch_size):
                idx = perm[start: start + self.batch_size]
                Xb, yb, sb = X[idx], y[idx], sample_weights[idx]
                m = len(idx)           # actual batch size (may differ at last batch)
                step += 1
                eta = self._eta(step, lam)

                scores = Xb @ w + b
                margins = yb * scores
                hinge = margins < 1.0

                # Regularisation shrink: (1 - eta*lambda) * w
                w *= (1.0 - eta * lam)

                if hinge.any():
                    coef = sb[hinge] * yb[hinge]            # (m_h,)
                    # Divide by m (batch size), NOT by |hinge|: unbiased gradient estimate
                    w += (eta / m) * (coef[:, None] * Xb[hinge]).sum(axis=0)
                    if self.fit_intercept:
                        b += (eta / m) * coef.sum()

                epoch_hinge += (sb * np.maximum(0.0, 1.0 - margins)).sum()

            loss = 0.5 * lam * w @ w + epoch_hinge / n
            losses.append(loss)
            if self.verbose and (epoch + 1) % log_every == 0:
                print(f"  [SGD] epoch {epoch+1}/{self.epochs}  loss={loss:.4f}")

        self.w_ = w
        self.b_ = b
        self.losses_ = losses

    # ------------------------------------------------------------------
    # Kernelized Pegasos (RBF)
    # ------------------------------------------------------------------

    def _fit_kernel_pegasos(self, X, y, sample_weights):
        """
        Budget-constrained Kernelized Pegasos for RBF SVM.

        Maintains a sparse representation:
            f(x) = Σ_{j ∈ S} coeff_j * y_j * K(x_j, x) + b

        At each step:
          1. Decay all coefficients by (1 - eta*lambda).
          2. For violated samples in the batch, add a new SV or accumulate.
          3. Prune SVs with |coeff| < threshold when budget is exceeded.
        """
        n, d = X.shape
        lam = 1.0 / self.C
        rng = np.random.default_rng(self.random_state)

        # Support vector storage (lists for dynamic growth)
        sv_X     = []          # list of (d,) arrays
        sv_coef  = []          # list of floats  (coeff_j, signed by y_j already absorbed below)
        sv_y     = []          # list of ±1

        b = 0.0
        losses = []
        step = 0
        log_every = max(1, self.epochs // 10)

        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            epoch_hinge = 0.0

            for start in range(0, n, self.batch_size):
                idx = perm[start: start + self.batch_size]
                Xb, yb, sb = X[idx], y[idx], sample_weights[idx]
                m = len(idx)
                step += 1
                eta = self._eta(step, lam)

                # ---- decay existing coefficients ----
                decay = 1.0 - eta * lam
                sv_coef = [c * decay for c in sv_coef]

                # ---- compute decision scores ----
                if sv_X:
                    SX = np.vstack(sv_X)                   # (|S|, d)
                    sa = np.array(sv_coef) * np.array(sv_y)  # (|S|,)
                    K_bS = self._rbf(Xb, SX)               # (m, |S|)
                    scores = K_bS @ sa + b
                else:
                    scores = np.full(m, b)

                margins = yb * scores
                hinge = margins < 1.0
                epoch_hinge += (sb * np.maximum(0.0, 1.0 - margins)).sum()

                # ---- add new support vectors for violated samples ----
                for j in np.where(hinge)[0]:
                    new_c = eta * float(sb[j]) / m
                    sv_X.append(Xb[j].copy())
                    sv_coef.append(new_c)
                    sv_y.append(float(yb[j]))

                # ---- update bias (divide by m, not |hinge|) ----
                if self.fit_intercept and hinge.any():
                    b += (eta / m) * (sb[hinge] * yb[hinge]).sum()

                # ---- budget pruning ----
                if self.budget is not None and len(sv_X) > self.budget:
                    abs_c = np.abs(sv_coef)
                    keep = np.argsort(abs_c)[-self.budget:]
                    sv_X    = [sv_X[i]    for i in keep]
                    sv_coef = [sv_coef[i] for i in keep]
                    sv_y    = [sv_y[i]    for i in keep]

            losses.append(epoch_hinge / n)  # approx (||w||^2 hard to compute)
            if self.verbose and (epoch + 1) % log_every == 0:
                print(f"  [Pegasos] epoch {epoch+1}/{self.epochs}  "
                      f"hinge={losses[-1]:.4f}  |S|={len(sv_X)}")

        # Store final model
        if sv_X:
            self.sv_X_    = np.vstack(sv_X)
            self.sv_coef_ = np.array(sv_coef)
            self.sv_y_    = np.array(sv_y)
        else:
            self.sv_X_    = np.zeros((0, d))
            self.sv_coef_ = np.zeros(0)
            self.sv_y_    = np.zeros(0)

        self.w_  = None
        self.b_  = b
        self.losses_ = losses

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def _class_balance_weights(self, y_label, base_weights):
        """
        Apply class-balanced scaling to base_weights.

        For each class c ∈ {-1, +1}:
            weight_c = n / (n_classes * count_c)
        This makes the total gradient contribution equal from both classes,
        compensating for the missing dual constraint Σ αᵢ yᵢ = 0.
        """
        if self.class_weight != 'balanced':
            return base_weights
        n = len(y_label)
        weights = base_weights.copy()
        for cls in np.unique(y_label):
            mask = y_label == cls
            weights[mask] *= n / (2.0 * mask.sum())
        return weights

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64, order='C')
        self._label_encoder = LabelEncoder()
        y_enc   = self._label_encoder.fit_transform(y)
        y_label = (2 * y_enc - 1).astype(np.float64)
        self.classes_ = self._label_encoder.classes_

        n = X.shape[0]
        sample_weights = self._class_balance_weights(y_label, np.ones(n))

        self._fit_with_weights(X, y_label, sample_weights)
        return self

    def _fit_with_weights(self, X, y_label, sample_weights):
        """Internal: dispatch to linear SGD or Kernelized Pegasos."""
        if self.kernel == 'rbf':
            self.gamma_ = self._resolve_gamma(X)
            self._fit_kernel_pegasos(X, y_label, sample_weights)
        elif self.kernel == 'linear':
            self._fit_linear_sgd(X, y_label, sample_weights)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel!r}. Use 'linear' or 'rbf'.")

    def decision_function(self, X):
        check_is_fitted(self, ['classes_'])
        X = check_array(X, accept_sparse=False, dtype=np.float64, order='C')

        if self.kernel == 'linear':
            return X @ self.w_ + self.b_
        else:
            if self.sv_X_.shape[0] == 0:
                return np.full(X.shape[0], self.b_)
            K = self._rbf(X, self.sv_X_)
            return K @ (self.sv_coef_ * self.sv_y_) + self.b_

    def predict(self, X):
        scores = self.decision_function(X)
        signs  = (scores >= 0).astype(int)
        return self._label_encoder.inverse_transform(signs)


# ---------------------------------------------------------------------------
# ProbSVM_SGD
# ---------------------------------------------------------------------------

class ProbSVM_SGD(NaiveSVM_SGD):
    """
    Probabilistic-weighted SVM with GD.

    Each sample's penalty is scaled by a per-sample weight s_i that encodes
    how "typical" the sample is for its class, via Mahalanobis-distance capping:

        D_M(x_i, c)  = sqrt( Σ_j (x_ij − μ_cj)² / σ²_cj )   (j ≤ k dims)
        τ_c          = 95th-percentile Mahalanobis radius of class c
        s_i          = clip( τ_c / D_M(x_i, c),  1e-6, 1.0 )

    This replaces the earlier log-Gaussian likelihood scheme.  Key benefits:

    • Clean accuracy preserved: ~95 % of clean samples satisfy D_M ≤ τ → s_i = 1
      (essentially identical to NaiveSVM on the majority of training points).
    • Noise robustness: Type I noise is generated with D_M > τ by construction,
      so those samples are automatically down-weighted.
    • No weight collapse: weights do not approach 0 in high dimensions — the
      Mahalanobis threshold is scale-free across any dimensionality.

    Additional Parameters
    ---------------------
    n_prob_components : int or None
        Number of leading features to use for Mahalanobis distance computation.
        None (default) → use all d features.
        Recommended: 8–32 for PCA-reduced data (focuses on highest-variance dims).
    """

    def __init__(
        self,
        C=1.0, fit_intercept=True, kernel='linear', gamma='scale',
        lr=0.01, lr_schedule='pegasos', epochs=100, batch_size=32,
        budget=300, verbose=False, random_state=42, class_weight='balanced',
        n_prob_components=None,
    ):
        super().__init__(
            C=C, fit_intercept=fit_intercept, kernel=kernel, gamma=gamma,
            lr=lr, lr_schedule=lr_schedule, epochs=epochs, batch_size=batch_size,
            budget=budget, verbose=verbose, random_state=random_state,
            class_weight=class_weight,
        )
        self.n_prob_components = n_prob_components

    def _compute_class_conditionals(self, X, y):
        """
        Compute per-sample weights via Mahalanobis-distance capping.

        For each class c:
            D_M(x, c) = sqrt( Σ_j (x_j − μ_cj)² / σ²_cj )  [j over first k dims]
            τ_c       = 95th-percentile of {D_M(x_i, c) : i ∈ class c}
            weight_i  = clip( τ_c / D_M(x_i, c),  1e-6,  1.0 )

        Why this beats the log-Gaussian scheme
        ----------------------------------------
        The old log-Gaussian approach produced *relative* weights via max-
        subtraction: the most-typical sample got 1.0 and everything else was
        exp(–Δlog_like).  For boundary / hard samples this meant low weights,
        which reduced the classifier's ability to learn the decision surface
        (lower clean accuracy than NaiveSVM).

        With Mahalanobis capping:
        • ~95 % of clean samples satisfy D_M ≤ τ → weight = 1.0 (same as NaiveSVM)
        • Type I noise (D_M >> τ by construction) → weight = τ/D_M << 1
        • Clean accuracy is preserved; Type I noise robustness is gained.

        y should be original multiclass labels when available (passed via
        y_original in fit) so each class mean/var is tight and unimodal.
        """
        k = self.n_prob_components
        if k is not None and k < X.shape[1]:
            Xw = X[:, :k]
        else:
            Xw = X

        classes   = np.unique(y)
        n_samples = len(y)
        weights   = np.ones(n_samples)

        for cls in classes:
            mask = y == cls
            if mask.sum() < 2:
                continue
            Xc  = Xw[mask]
            mu  = Xc.mean(axis=0)
            var = Xc.var(axis=0) + 1e-9

            D_sq = ((Xc - mu) ** 2 / var).sum(axis=1)
            D    = np.sqrt(np.maximum(D_sq, 0.0))

            # τ: 95th-percentile Mahalanobis radius of clean class distribution
            tau = np.percentile(D, 95) + 1e-9
            weights[mask] = np.clip(tau / (D + 1e-9), 1e-6, 1.0)

        return weights

    def fit(self, X, y, y_original=None):
        """
        Parameters
        ----------
        X          : feature matrix
        y          : binary labels {0,1} (from OneVsRest) or multiclass
        y_original : original multiclass labels before OneVsRest binarisation.
                     When provided, Gaussian weights are computed per original
                     class (tight unimodal Gaussians) rather than fitting one
                     Gaussian to the heterogeneous "rest" mixture.
                     Passed automatically by OneVsRestSVM.
        """
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64, order='C')
        self._label_encoder = LabelEncoder()
        y_enc   = self._label_encoder.fit_transform(y)
        y_label = (2 * y_enc - 1).astype(np.float64)
        self.classes_ = self._label_encoder.classes_

        p_i = self._compute_class_conditionals(
            X, y_original if y_original is not None else y_label
        )
        p_i = self._class_balance_weights(y_label, p_i)
        self.p_i_ = p_i

        self._fit_with_weights(X, y_label, p_i)
        return self


# ---------------------------------------------------------------------------
# ANNSVM_SGD  (Approximate Nearest-Neighbor weighted SVM via FAISS)
# ---------------------------------------------------------------------------

class ANNSVM_SGD(NaiveSVM_SGD):
    """
    ANN-agreement-weighted SVM trained with GD.

    Each sample's penalty is scaled by w_i — the inverse-distance-weighted
    fraction of its k approximate nearest neighbors that share the same label.
    Neighborhood is found with FAISS (exact flat index for n < 10k, IVF
    approximate for larger datasets), making this practical on CIFAR-10 scale.

        s_i = w_i ∈ (0, 1]   (higher → more "typical" point, stronger SV)

    Additional Parameters
    ---------------------
    k : int
        Number of approximate nearest neighbors to query (default 5).
    n_ann_components : int or None
        Number of leading PCA components to use for ANN distance computation.
        None (default) → use all features.
        Recommended: 8–32 for PCA-reduced data to mitigate the curse of
        dimensionality — high-dimensional L2 distances become uninformative,
        while top PCA components capture the most class-discriminative variance.
    """

    def __init__(self, C=1.0, fit_intercept=True, kernel='linear', gamma='scale',
                 lr=0.01, lr_schedule='pegasos', epochs=100, batch_size=32,
                 budget=300, verbose=False, random_state=42, class_weight='balanced',
                 k=5, n_ann_components=None):
        super().__init__(
            C=C, fit_intercept=fit_intercept, kernel=kernel, gamma=gamma,
            lr=lr, lr_schedule=lr_schedule, epochs=epochs, batch_size=batch_size,
            budget=budget, verbose=verbose, random_state=random_state,
            class_weight=class_weight,
        )
        self.k = k
        self.n_ann_components = n_ann_components

    def _compute_ann_weights(self, X, y):
        """
        Build FAISS index, query k ANN for each point, compute w_i.

        w_i = Σ_{j∈same-label} (1/d_j) / Σ_j (1/d_j)

        ANN search is performed in X[:, :n_ann_components] (top PCA components)
        to avoid the curse of dimensionality in high-dimensional spaces.
        """
        n = X.shape[0]
        k = self.n_ann_components
        X_search = X[:, :k] if (k is not None and k < X.shape[1]) else X
        X_f32 = np.ascontiguousarray(X_search, dtype=np.float32)

        if FAISS_AVAILABLE:
            self.ann_index_ = _build_faiss_index(X_f32, self.k)
            distances, indices = _ann_kneighbors(self.ann_index_, X_f32, self.k)
        else:
            distances, indices = _sklearn_kneighbors(X, self.k)

        w_i = np.zeros(n)
        for i in range(n):
            nb_labels = y[indices[i]]
            dists     = distances[i]
            same      = nb_labels == y[i]

            inv_d = 1.0 / (dists + 1e-9)
            total = inv_d.sum()
            w_i[i] = inv_d[same].sum() / total if total > 0 else same.sum() / self.k

        return w_i

    def fit(self, X, y, y_original=None):
        """
        Parameters
        ----------
        y_original : original multiclass labels (before OneVsRest binarisation).
                     When provided, ANN agreement is measured against the original
                     per-class labels instead of binary ±1.  This prevents the
                     "rest" class inflation problem in OvR: samples at the boundary
                     between two different rest classes would otherwise both be
                     labelled -1 and get w_i≈1.0, artificially inflating their
                     weights.  Passed automatically by OneVsRestSVM.
        """
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64, order='C')
        self._label_encoder = LabelEncoder()
        y_enc   = self._label_encoder.fit_transform(y)
        y_label = (2 * y_enc - 1).astype(np.float64)
        self.classes_ = self._label_encoder.classes_

        y_for_ann = y_original if y_original is not None else y
        w_i = self._compute_ann_weights(X, y_for_ann)
        w_i = self._class_balance_weights(y_label, w_i)
        self.w_i_ = w_i

        self._fit_with_weights(X, y_label, w_i)
        return self


# ---------------------------------------------------------------------------
# SKiP_SGD  (ANN + Probabilistic, combined)
# ---------------------------------------------------------------------------

class SKiP_SGD(ANNSVM_SGD):
    """
    SKiP with Gradient Descent + FAISS ANN.

    Combines ANN-agreement weights (w_i via FAISS) and class-conditional
    probability weights (p_i via log-space Gaussian likelihood):

        s_i = combine(w_i, p_i)

    Additional Parameters
    ---------------------
    combine_method : {'average', 'multiply'}
        How to fuse w_i and p_i.
        'average' (default): s_i = 0.5*(w_i + p_i).  Because p_i ≈ 1 for ~95%
            of clean samples (Mahalanobis capping), s_i ≈ 0.5*(w_i+1) which
            pulls ANN weights toward 1 — preserving clean accuracy while still
            down-weighting noisy samples from both signals.
        'multiply': s_i = w_i * p_i.  Stronger noise suppression but can also
            suppress important boundary samples, reducing clean accuracy.
    scaling : {None, 'minmax'}
        Optional min-max normalisation of w_i and p_i before combining.
    """

    def __init__(self, C=1.0, fit_intercept=True, kernel='linear', gamma='scale',
                 lr=0.01, lr_schedule='pegasos', epochs=100, batch_size=32,
                 budget=300, verbose=False, random_state=42, class_weight='balanced',
                 k=5, combine_method='average', scaling=None, n_prob_components=None,
                 n_ann_components=None):
        super().__init__(
            C=C, fit_intercept=fit_intercept, kernel=kernel, gamma=gamma,
            lr=lr, lr_schedule=lr_schedule, epochs=epochs, batch_size=batch_size,
            budget=budget, verbose=verbose, random_state=random_state,
            class_weight=class_weight, k=k, n_ann_components=n_ann_components,
        )
        self.combine_method    = combine_method
        self.scaling           = scaling
        self.n_prob_components = n_prob_components

    # Mahalanobis-capping weight computation — same scheme as ProbSVM_SGD
    def _compute_class_conditionals(self, X, y):
        """Mahalanobis-distance-capped weights (see ProbSVM_SGD for full docstring)."""
        kk = self.n_prob_components
        Xw = X[:, :kk] if (kk is not None and kk < X.shape[1]) else X

        classes   = np.unique(y)
        n_samples = len(y)
        weights   = np.ones(n_samples)

        for cls in classes:
            mask = y == cls
            if mask.sum() < 2:
                continue
            Xc  = Xw[mask]
            mu  = Xc.mean(axis=0)
            var = Xc.var(axis=0) + 1e-9

            D_sq = ((Xc - mu) ** 2 / var).sum(axis=1)
            D    = np.sqrt(np.maximum(D_sq, 0.0))

            tau = np.percentile(D, 95) + 1e-9
            weights[mask] = np.clip(tau / (D + 1e-9), 1e-6, 1.0)

        return weights

    def _combine(self, w_i, p_i):
        if self.scaling == 'minmax':
            def minmax(v):
                lo, hi = v.min(), v.max()
                return np.clip((v - lo) / (hi - lo + 1e-9), 1e-6, 1.0)
            w_i, p_i = minmax(w_i), minmax(p_i)

        if self.combine_method == 'multiply':
            return w_i * p_i
        if self.combine_method == 'average':
            return 0.5 * (w_i + p_i)
        raise ValueError(f"Unknown combine_method: {self.combine_method!r}")

    def fit(self, X, y, y_original=None):
        """
        Parameters
        ----------
        y_original : original multiclass labels (before OneVsRest binarisation).
                     When provided, class-conditional Gaussians are fitted per
                     original class → tighter, unimodal distributions → better
                     weight discrimination. Passed automatically by OneVsRestSVM.
        """
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64, order='C')
        self._label_encoder = LabelEncoder()
        y_enc   = self._label_encoder.fit_transform(y)
        y_label = (2 * y_enc - 1).astype(np.float64)
        self.classes_ = self._label_encoder.classes_

        w_i = self._compute_ann_weights(X, y_label)
        # Use original multiclass labels for Gaussian fitting if available
        p_i = self._compute_class_conditionals(
            X, y_original if y_original is not None else y_label
        )
        combined = self._combine(w_i, p_i)
        combined = self._class_balance_weights(y_label, combined)

        self.w_i_             = w_i
        self.p_i_             = p_i
        self.combined_weight_ = combined

        self._fit_with_weights(X, y_label, combined)
        return self


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

#: ``KNNSVM_GD`` was renamed to :class:`ANNSVM_SGD` to reflect that the
#: neighbor search is *approximate* (FAISS) rather than exact KNN.
#: The alias keeps old code working without modification.
KNNSVM_GD = ANNSVM_SGD
