import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
import cvxpy as cp

class NaiveSVM(BaseEstimator, ClassifierMixin):
    """
    Linear soft-margin SVM trained with simple SGD on the primal:
        minimize 0.5 * ||w||^2 + C * sum_i hinge(1 - y_i * (w^T x_i + b))
    where y_i in {-1, +1}.
    """

    def __init__(self, C=1.0, fit_intercept=True, verbose=False, tol=1e-6):
        self.C = float(C)
        self.fit_intercept = bool(fit_intercept)
        self.verbose = bool(verbose)
        self.tol = float(tol)

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64, order='C')

        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y)          # {0,1}
        y_label = (2 * y_enc - 1).astype(np.float64)             # {-1,+1} labeling
        self.classes_ = self._label_encoder.classes_

        n, d = X.shape
        C = float(self.C)

        # Dual QP configuration
        K = X @ X.T                                           # (n,n) 
        Q = (y_label[:, None] * y_label[None, :]) * K         # (n,n) -> yiyjKij
        # Q = 0.5 * (Q + Q.T)
        # Q += 1e-12 * np.eye(n)

        alpha = cp.Variable(n)

        # Objective: min 0.5 * a^T Q a - 1^T a (cvpyx only supports min optimization)
        objective = cp.Minimize(0.5 * cp.quad_form(alpha, cp.psd_wrap(Q)) - cp.sum(alpha))

        # constraint
        constraints = [alpha >= 0, alpha <= C]
        if self.fit_intercept:
            constraints.append(cp.sum(cp.multiply(alpha, y_label)) == 0)

        prob = cp.Problem(objective, constraints)

        prob.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, verbose=self.verbose)

        a = np.asarray(alpha.value).ravel()
        a = np.clip(a, 0.0, C)

        self.w_ = (a * y_label) @ X                              # (d,)
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
            self.support_ = np.where(a > self.tol)[0].astype(int)

        self.alphas_ = a

        return self

    def decision_function(self, X):
        check_is_fitted(self, attributes=["w_", "b_", "classes_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64, order='C')
        return X @ self.w_ + (self.b_ if self.fit_intercept else 0.0)

    def predict(self, X):
        scores = self.decision_function(X)
        # Map sign back to original labels
        signs = (scores >= 0).astype(int)  # 0 for -1, 1 for +1
        # Reverse of y_label = 2*y_enc - 1 -> y_enc = (y_label+1)/2
        return self._label_encoder.inverse_transform(signs)


class ProbSVM(BaseEstimator, ClassifierMixin):
    
    def __init__(self, C=1.0, fit_intercept=True, verbose=False, tol=1e-6):
        self.C = float(C)
        self.fit_intercept = bool(fit_intercept)
        self.verbose = bool(verbose)
        self.tol = float(tol)

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
        K = X @ X.T
        Q = (y_label[:, None] * y_label[None, :]) * K
        Q = 0.5 * (Q + Q.T) + 1e-9 * np.eye(n)

        alpha = cp.Variable(n)
        try:
            objective = cp.Minimize(0.5 * cp.quad_form(alpha, Q) - cp.sum(alpha))

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
        except cp.error.DCPError:
            # Use psd_wrap to ensure Q is treated as PSD for DCP compliance
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

        # Compute weight vector
        self.w_ = (a * y_label) @ X

        # Compute bias
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
            self.support_ = np.where(a > self.tol)[0].astype(int)

        self.alphas_ = a
        self.p_i_ = p_i

        return self

    def decision_function(self, X):
        check_is_fitted(self, attributes=["w_", "b_", "classes_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64, order='C')
        return X @ self.w_ + (self.b_ if self.fit_intercept else 0.0)

    def predict(self, X):
        scores = self.decision_function(X)
        # Map sign back to original labels
        signs = (scores >= 0).astype(int)  # 0 for -1, 1 for +1
        # Reverse of y_label = 2*y_enc - 1 -> y_enc = (y_label+1)/2
        return self._label_encoder.inverse_transform(signs)
