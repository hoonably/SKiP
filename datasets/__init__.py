import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler

def precompute_class_stats(X_train, y_train, percentile=99):
    class_stats = {}
    classes = np.unique(y_train)

    for cls in classes:
        X_cls = X_train[y_train == cls]

        mu = X_cls.mean(axis=0)
        cov = np.cov(X_cls.T) + np.eye(X_train.shape[1]) * 1e-6

        L = np.linalg.cholesky(cov)

        inv_cov = np.linalg.inv(cov)
        diffs = X_cls - mu
        dists = np.einsum('ni,ij,nj->n', diffs, inv_cov, diffs)
        tau = np.percentile(dists, percentile)

        class_stats[cls] = {
            "mu": mu,
            "L": L,
            "tau": tau,
        }

    return class_stats


def inject_noise(X_train, y_train, feature_noise=0.0, label_noise=0.0, random_state=42, add_label_noise=False, epsilon=1e-3):
    rng = default_rng(random_state)
    X = X_train.copy()
    y = y_train.copy()
    d = X.shape[1]


    if label_noise > 0:
        n_label_flips = int(len(y) * label_noise)
        flip_indices = rng.choice(len(y), n_label_flips, replace=False)
        unique_labels = np.unique(y)

        if add_label_noise is False:
            for idx in flip_indices:
                original = y[idx]
                y[idx] = rng.choice(unique_labels[unique_labels != original])

        else:
            cov_feat = np.cov(X_train, rowvar=False) + np.eye(d) * 1e-12
            feature_std = np.sqrt(np.diag(cov_feat))
            new_X = []
            new_y = []

            for idx in flip_indices:
                original_label = y[idx]
                flipped_label = rng.choice(unique_labels[unique_labels != original_label])

                base_noise = rng.standard_normal(size=d)
                noise = epsilon * feature_std * base_noise
                x_noisy = X[idx] + noise

                new_X.append(x_noisy)
                new_y.append(flipped_label)

            new_X = np.vstack(new_X)
            new_y = np.array(new_y)

            X = np.vstack([X, new_X])
            y = np.concatenate([y, new_y])


    if feature_noise > 0:
        class_stats = precompute_class_stats(X_train, y_train, percentile=99)

        n_feature_outliers = int(len(X) * feature_noise)
        outliers = []
        outlier_labels = []

        classes = np.array(list(class_stats.keys()))
        d = X_train.shape[1]

        for _ in range(n_feature_outliers):
            cls = rng.choice(classes)
            stats = class_stats[cls]
            mu = stats["mu"]
            L = stats["L"]
            tau = stats["tau"]

            z = rng.standard_normal(size=d)

            mah_distance = np.dot(z, z)
            E = rng.exponential(scale= 2 / tau)
            r2 = tau + E

            scale = np.sqrt(r2 / (mah_distance + 1e-12))
            z = z * scale

            x_candidate = mu + L @ z

            outliers.append(x_candidate)
            outlier_labels.append(cls)

        outliers = np.array(outliers)
        outlier_labels = np.array(outlier_labels)

        X = np.vstack([X, outliers])
        y = np.concatenate([y, outlier_labels])

    return X, y