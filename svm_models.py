import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import cvxpy as cp


class NaiveSVM(BaseEstimator, ClassifierMixin):
    """
    Linear soft-margin SVM trained with simple SGD on the primal:
        minimize 0.5 * ||w||^2 + C * sum_i hinge(1 - y_i * (w^T x_i + b))
    where y_i in {-1, +1}.
    """

    def __init__(self, C=1.0, fit_intercept=True, verbose=False, tol=1e-6, kernel='linear', gamma='scale'):
        self.C = float(C)
        self.fit_intercept = bool(fit_intercept)
        self.verbose = bool(verbose)
        self.tol = float(tol)
        self.kernel = kernel
        self.gamma = gamma

    def _compute_kernel(self, X1, X2=None):
        """Compute kernel matrix between X1 and X2."""
        if X2 is None:
            X2 = X1

        if self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'rbf':
            if self.gamma == 'scale':
                gamma_value = 1.0 / (X1.shape[1] * X1.var())
            elif self.gamma == 'auto':
                gamma_value = 1.0 / X1.shape[1]
            else:
                gamma_value = float(self.gamma)

            # Compute squared Euclidean distances
            X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
            distances_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
            return np.exp(-gamma_value * distances_sq)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64, order='C')

        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y)          # {0,1}
        y_label = (2 * y_enc - 1).astype(np.float64)             # {-1,+1} labeling
        self.classes_ = self._label_encoder.classes_

        n, d = X.shape
        C = float(self.C)

        # Dual QP configuration
        K = self._compute_kernel(X)                           # (n,n) 
        Q = (y_label[:, None] * y_label[None, :]) * K         # (n,n) -> yiyjKij
        Q = 0.5 * (Q + Q.T) + 1e-10 * np.eye(n)  # Symmetrize and add regularization

        alpha = cp.Variable(n)

        # Objective: min 0.5 * a^T Q a - 1^T a (cvpyx only supports min optimization)
        # Use psd_wrap for numerical stability, especially with RBF kernel
        objective = cp.Minimize(0.5 * cp.quad_form(alpha, cp.psd_wrap(Q)) - cp.sum(alpha))

        # constraint
        constraints = [alpha >= 0, alpha <= C]
        if self.fit_intercept:
            constraints.append(cp.sum(cp.multiply(alpha, y_label)) == 0)

        prob = cp.Problem(objective, constraints)

        prob.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, verbose=self.verbose)

        a = np.asarray(alpha.value).ravel()
        a = np.clip(a, 0.0, C)

        # For linear kernel, compute explicit weight vector
        if self.kernel == 'linear':
            self.w_ = (a * y_label) @ X                              # (d,)
        else:
            self.w_ = None  # No explicit weight vector for RBF kernel

        # Compute bias and identify support vectors
        if self.fit_intercept:
            tol = float(self.tol)
            sv_all  = np.where(a > tol)[0]
            sv_free = np.where((a > tol) & (a < C - tol))[0]
            idx = sv_free if sv_free.size > 0 else sv_all
            if idx.size > 0:
                decision_at_sv = (a * y_label) @ K[:, idx]       # (|idx|,)
                b_vals = y_label[idx] - decision_at_sv
                self.b_ = float(np.mean(b_vals))
            else:
                self.b_ = 0.0
            self.support_ = sv_all.astype(int)
        else:
            self.b_ = 0.0
            sv_all = np.where(a > self.tol)[0]
            self.support_ = sv_all.astype(int)

        # Store only support vectors for prediction
        if len(self.support_) > 0:
            self.X_sv_ = X[self.support_].copy()
            self.alphas_sv_ = a[self.support_]
            self.y_sv_ = y_label[self.support_]
            if self.verbose:
                print(f"Support vectors: {len(self.support_)}/{n} ({100*len(self.support_)/n:.1f}%)")
        else:
            # Fallback: use all samples if no support vectors found
            self.X_sv_ = X.copy()
            self.alphas_sv_ = a
            self.y_sv_ = y_label

        return self

    def decision_function(self, X):
        check_is_fitted(self, attributes=["alphas_sv_", "y_sv_", "X_sv_", "classes_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64, order='C')

        if self.kernel == 'linear':
            return X @ self.w_ + (self.b_ if self.fit_intercept else 0.0)
        else:
            # For kernel methods, compute decision using kernel and support vectors only
            K = self._compute_kernel(X, self.X_sv_)
            return (self.alphas_sv_ * self.y_sv_) @ K.T + (self.b_ if self.fit_intercept else 0.0)

    def predict(self, X):
        scores = self.decision_function(X)
        # Map sign back to original labels
        signs = (scores >= 0).astype(int)  # 0 for -1, 1 for +1
        # Reverse of y_label = 2*y_enc - 1 -> y_enc = (y_label+1)/2
        return self._label_encoder.inverse_transform(signs)


class ProbSVM(NaiveSVM):

    def __init__(self, C=1.0, fit_intercept=True, verbose=False, tol=1e-6, kernel='linear', gamma='scale'):
        super().__init__(C=C, fit_intercept=fit_intercept, verbose=verbose, tol=tol, kernel=kernel, gamma=gamma)

    def _gaussian_prob(self, X, mean, var):
        return np.exp(-0.5 * ((X - mean) ** 2) / var) / np.sqrt(2.0 * np.pi * var)

    def _compute_class_conditionals(self, X, y):
        classes = np.unique(y)
        n_samples = len(y)
        p_i = np.zeros(n_samples)

        for cls in classes:
            cls_mask = (y == cls)
            prior = np.sum(cls_mask) / n_samples

            X_cls = X[cls_mask]
            mean = np.mean(X_cls, axis=0)
            var = np.var(X_cls, axis=0) + 1e-9

            # likelihood for all samples given this class
            likelihood = np.prod(self._gaussian_prob(X_cls, mean, var), axis=1)

            # only use the likelihood corresponding to the true label
            p_i[cls_mask] = prior * likelihood

        # Normalize so that weights aren't absurdly large/small
        p_i /= np.max(p_i)
        return np.clip(p_i, 1e-6, 1.0)

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64, order='C')

        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y)
        y_label = (2 * y_enc - 1).astype(np.float64)
        self.classes_ = self._label_encoder.classes_

        n, d = X.shape
        C = float(self.C)

        # Compute probabilistic weights
        p_i = self._compute_class_conditionals(X, y_label)

        # Dual QP configuration
        K = self._compute_kernel(X)
        Q = (y_label[:, None] * y_label[None, :]) * K
        Q = 0.5 * (Q + Q.T) + 1e-10 * np.eye(n)  # Symmetrize and add regularization

        alpha = cp.Variable(n)
        # Use psd_wrap from the start for numerical stability, especially with RBF kernel
        objective = cp.Minimize(0.5 * cp.quad_form(alpha, cp.psd_wrap(Q)) - cp.sum(alpha))

        constraints = [alpha >= 0, alpha <= C * p_i]
        if self.fit_intercept:
            constraints.append(cp.sum(cp.multiply(alpha, y_label)) == 0)

        prob = cp.Problem(objective, constraints)
        prob.solve(
            solver=cp.OSQP,
            eps_abs=1e-6,
            eps_rel=1e-6,
            verbose=self.verbose
        )

        a = np.asarray(alpha.value).ravel()
        a = np.clip(a, 0.0, C * p_i)

        # Compute weight vector (only for linear kernel)
        if self.kernel == 'linear':
            self.w_ = (a * y_label) @ X
        else:
            self.w_ = None

        # Compute bias and identify support vectors
        if self.fit_intercept:
            tol = float(self.tol)
            sv_all  = np.where(a > tol)[0]
            sv_free = np.where((a > tol) & (a < C * p_i - tol))[0]
            idx = sv_free if sv_free.size > 0 else sv_all
            if idx.size > 0:
                decision = (a * y_label) @ K
                b_vals = y_label[idx] - decision[idx]
                self.b_ = np.mean(b_vals)
            else:
                self.b_ = 0.0
            self.support_ = sv_all.astype(int)
        else:
            self.b_ = 0.0
            sv_all = np.where(a > self.tol)[0]
            self.support_ = sv_all.astype(int)

        # Store only support vectors for prediction
        if len(self.support_) > 0:
            self.X_sv_ = X[self.support_].copy()
            self.alphas_sv_ = a[self.support_]
            self.y_sv_ = y_label[self.support_]
            if self.verbose:
                print(f"Support vectors: {len(self.support_)}/{n} ({100*len(self.support_)/n:.1f}%)")
        else:
            # Fallback: use all samples if no support vectors found
            self.X_sv_ = X.copy()
            self.alphas_sv_ = a
            self.y_sv_ = y_label

        self.p_i_ = p_i

        return self


class KNNSVM(NaiveSVM):

    def __init__(self, C=1.0, fit_intercept=True, verbose=False, tol=1e-6, k=5, metric='euclidean', kernel='linear', gamma='scale'):
        super().__init__(C=C, fit_intercept=fit_intercept, verbose=verbose, tol=tol, kernel=kernel, gamma=gamma)
        self.k = int(k)
        self.metric = metric

    def _compute_knn_weights(self, X, y):

        n_samples = X.shape[0]
        distances, indices = self.nn_.kneighbors(X)

        # Exclude the point itself (first neighbor)
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        w_i = np.zeros(n_samples)

        for i in range(n_samples):
            neighbor_labels = y[indices[i]]
            neighbor_distances = distances[i]

            # Count same-label neighbors
            same_label_mask = (neighbor_labels == y[i])
            n_same = np.sum(same_label_mask)

            # Weight based on label agreement and inverse distance
            # Higher weight if more neighbors share the same label
            # and neighbors are closer
            if np.sum(neighbor_distances) > 0:
                inv_distances = 1.0 / (neighbor_distances + 1e-9)
                same_label_contribution = np.sum(inv_distances[same_label_mask])
                total_contribution = np.sum(inv_distances)
                w_i[i] = same_label_contribution / total_contribution
            else:
                w_i[i] = n_same / self.k

        # Normalize so that weights aren't absurdly large/small
        w_i = np.clip(w_i, 1e-6, 1.0)

        return w_i

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64, order='C')

        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y)
        y_label = (2 * y_enc - 1).astype(np.float64)
        self.classes_ = self._label_encoder.classes_

        n, d = X.shape
        C = float(self.C)

        # Store training data for kernel methods
        self.X_train_ = X.copy()

        # Initialize and fit NearestNeighbors once
        self.nn_ = NearestNeighbors(n_neighbors=self.k + 1, metric=self.metric)
        self.nn_.fit(X)

        # Compute KNN-based weights
        w_i = self._compute_knn_weights(X, y_label)

        # Dual QP configuration
        K = self._compute_kernel(X)
        Q = (y_label[:, None] * y_label[None, :]) * K
        Q = 0.5 * (Q + Q.T) + 1e-10 * np.eye(n)  # Symmetrize and add regularization

        alpha = cp.Variable(n)

        objective = cp.Minimize(0.5 * cp.quad_form(alpha, cp.psd_wrap(Q)) - cp.sum(alpha))

        constraints = [alpha >= 0, alpha <= C * w_i]
        if self.fit_intercept:
            constraints.append(cp.sum(cp.multiply(alpha, y_label)) == 0)

        prob = cp.Problem(objective, constraints)
        prob.solve(
            solver=cp.OSQP,
            eps_abs=1e-6,
            eps_rel=1e-6,
            verbose=self.verbose
        )

        a = np.asarray(alpha.value).ravel()
        a = np.clip(a, 0.0, C * w_i)

        # Compute weight vector (only for linear kernel)
        if self.kernel == 'linear':
            self.w_ = (a * y_label) @ X
        else:
            self.w_ = None

        # Compute bias and identify support vectors
        if self.fit_intercept:
            tol = float(self.tol)
            sv_all  = np.where(a > tol)[0]
            sv_free = np.where((a > tol) & (a < C * w_i - tol))[0]
            idx = sv_free if sv_free.size > 0 else sv_all
            if idx.size > 0:
                decision = (a * y_label) @ K
                b_vals = y_label[idx] - decision[idx]
                self.b_ = np.mean(b_vals)
            else:
                self.b_ = 0.0
            self.support_ = sv_all.astype(int)
        else:
            self.b_ = 0.0
            sv_all = np.where(a > self.tol)[0]
            self.support_ = sv_all.astype(int)

        # Store only support vectors for prediction
        if len(self.support_) > 0:
            self.X_sv_ = X[self.support_].copy()
            self.alphas_sv_ = a[self.support_]
            self.y_sv_ = y_label[self.support_]
            if self.verbose:
                print(f"Support vectors: {len(self.support_)}/{n} ({100*len(self.support_)/n:.1f}%)")
        else:
            # Fallback: use all samples if no support vectors found
            self.X_sv_ = X.copy()
            self.alphas_sv_ = a
            self.y_sv_ = y_label

        self.w_i_ = w_i

        return self


class SKiP(NaiveSVM):

    def __init__(self, C=1.0, fit_intercept=True, verbose=False, tol=1e-6, k=5, metric='euclidean', combine_method='multiply', scaling='minmax', kernel='linear', gamma='scale'):
        super().__init__(C=C, fit_intercept=fit_intercept, verbose=verbose, tol=tol, kernel=kernel, gamma=gamma)
        self.k = int(k)
        self.metric = metric
        self.combine_method = combine_method
        self.scaling = scaling

    def _gaussian_prob(self, X, mean, var):
        return np.exp(-0.5 * ((X - mean) ** 2) / var) / np.sqrt(2.0 * np.pi * var)

    def _compute_class_conditionals(self, X, y):
        classes = np.unique(y)
        n_samples = len(y)
        p_i = np.zeros(n_samples)

        for cls in classes:
            cls_mask = (y == cls)
            prior = np.sum(cls_mask) / n_samples

            X_cls = X[cls_mask]
            mean = np.mean(X_cls, axis=0)
            var = np.var(X_cls, axis=0) + 1e-9

            # likelihood for all samples given this class
            likelihood = np.prod(self._gaussian_prob(X_cls, mean, var), axis=1)

            # only use the likelihood corresponding to the true label
            p_i[cls_mask] = prior * likelihood

        # Normalize so that weights aren't absurdly large/small
        p_i /= np.max(p_i)
        return np.clip(p_i, 1e-6, 1.0)

    def _compute_knn_weights(self, X, y):

        n_samples = X.shape[0]
        distances, indices = self.nn_.kneighbors(X)

        # Exclude the point itself (first neighbor)
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        w_i = np.zeros(n_samples)

        for i in range(n_samples):
            neighbor_labels = y[indices[i]]
            neighbor_distances = distances[i]

            # Count same-label neighbors
            same_label_mask = (neighbor_labels == y[i])
            n_same = np.sum(same_label_mask)

            # Weight based on label agreement and inverse distance
            # Higher weight if more neighbors share the same label
            # and neighbors are closer
            if np.sum(neighbor_distances) > 0:
                inv_distances = 1.0 / (neighbor_distances + 1e-9)
                same_label_contribution = np.sum(inv_distances[same_label_mask])
                total_contribution = np.sum(inv_distances)
                w_i[i] = same_label_contribution / total_contribution
            else:
                w_i[i] = n_same / self.k

        # Normalize so that weights aren't absurdly large/small
        w_i = np.clip(w_i, 1e-6, 1.0)

        return w_i

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64, order='C')

        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y)
        y_label = (2 * y_enc - 1).astype(np.float64)
        self.classes_ = self._label_encoder.classes_

        n, d = X.shape
        C = float(self.C)

        # Initialize and fit NearestNeighbors once
        self.nn_ = NearestNeighbors(n_neighbors=self.k + 1, metric=self.metric)
        self.nn_.fit(X)

        # Compute KNN weight, conditional weight
        w_i = self._compute_knn_weights(X, y_label)
        p_i = self._compute_class_conditionals(X, y_label)

        # Dual QP configuration
        K = self._compute_kernel(X)
        Q = (y_label[:, None] * y_label[None, :]) * K
        Q = 0.5 * (Q + Q.T) + 1e-10 * np.eye(n)  # Symmetrize and add regularization

        alpha = cp.Variable(n)

        objective = cp.Minimize(0.5 * cp.quad_form(alpha, cp.psd_wrap(Q)) - cp.sum(alpha))

        if self.scaling == 'minmax':
            w_i = (w_i - np.min(w_i)) / (np.max(w_i) - np.min(w_i) + 1e-9)
            p_i = (p_i - np.min(p_i)) / (np.max(p_i) - np.min(p_i) + 1e-9)
            w_i = np.clip(w_i, 1e-6, 1.0)
            p_i = np.clip(p_i, 1e-6, 1.0)

        if self.combine_method == 'multiply':
            combined_weight = w_i * p_i
        elif self.combine_method == 'average':
            combined_weight = 0.5 * (w_i + p_i)
        else:
            raise ValueError(f"Unknown combine_method: {self.combine_method}")

        constraints = [alpha >= 0, alpha <= C * combined_weight]
        if self.fit_intercept:
            constraints.append(cp.sum(cp.multiply(alpha, y_label)) == 0)

        prob = cp.Problem(objective, constraints)
        prob.solve(
            solver=cp.OSQP,
            eps_abs=1e-6,
            eps_rel=1e-6,
            verbose=self.verbose
        )

        a = np.asarray(alpha.value).ravel()
        a = np.clip(a, 0.0, C * combined_weight)

        # Compute weight vector (only for linear kernel)
        if self.kernel == 'linear':
            self.w_ = (a * y_label) @ X
        else:
            self.w_ = None

        # Compute bias and identify support vectors
        if self.fit_intercept:
            tol = float(self.tol)
            sv_all  = np.where(a > tol)[0]
            sv_free = np.where((a > tol) & (a < C * combined_weight - tol))[0]
            idx = sv_free if sv_free.size > 0 else sv_all
            if idx.size > 0:
                decision = (a * y_label) @ K
                b_vals = y_label[idx] - decision[idx]
                self.b_ = np.mean(b_vals)
            else:
                self.b_ = 0.0
            self.support_ = sv_all.astype(int)
        else:
            self.b_ = 0.0
            sv_all = np.where(a > self.tol)[0]
            self.support_ = sv_all.astype(int)

        # Store only support vectors for prediction
        if len(self.support_) > 0:
            self.X_sv_ = X[self.support_].copy()
            self.alphas_sv_ = a[self.support_]
            self.y_sv_ = y_label[self.support_]
            if self.verbose:
                print(f"Support vectors: {len(self.support_)}/{n} ({100*len(self.support_)/n:.1f}%)")
        else:
            # Fallback: use all samples if no support vectors found
            self.X_sv_ = X.copy()
            self.alphas_sv_ = a
            self.y_sv_ = y_label

        self.w_i_ = w_i
        self.p_i_ = p_i
        self.combined_weight_ = combined_weight

        return self
