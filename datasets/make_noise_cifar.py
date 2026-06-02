import numpy as np

# Outlier injection for CIFAR-10 PCA datasets.

def precompute_class_stats(X: np.ndarray, y: np.ndarray) -> dict:

    stats = {}
    for c in np.unique(y):
        Xc  = X[y == c]
        mu  = Xc.mean(axis=0)
        var = Xc.var(axis=0) + 1e-10 

        D   = np.sqrt(((Xc - mu) ** 2 / var).sum(axis=1))
        tau = np.percentile(D, 99)

        stats[c] = dict(mu=mu, var=var, tau=tau, D=D)

    return stats


def _generate_type1_sample(
    mu: np.ndarray,
    var: np.ndarray,
    tau: float,
    rng: np.random.Generator,
    margin: float = 0.5,
) -> np.ndarray:

    d = len(mu)

    u = rng.standard_normal(d)
    u /= np.linalg.norm(u) + 1e-12

    r = tau * (1.0 + rng.exponential(margin))

    x_noise = mu + r * u * np.sqrt(var)

    return x_noise


def inject_noise_cifar(
    X: np.ndarray,
    y: np.ndarray,
    type1_frac: float = 0.0,
    type2_frac: float = 0.0,
    type1_margin: float = 0.5,
    random_state: int = 42,
    stats: dict = None,
    verbose: bool = True,
) -> tuple:

    rng = np.random.default_rng(random_state)
    n   = len(y)

    assert 0.0 <= type1_frac, "type1_frac must be >= 0"
    assert 0.0 <= type2_frac, "type2_frac must be >= 0"

    n_type1 = int(n * type1_frac)
    n_type2 = int(n * type2_frac)

    X_parts = [X]
    y_parts = [y]

    # Feature Outliers

    if n_type1 > 0:
        if stats is None:
            if verbose:
                print('  [make_noise_cifar] Computing class stats for Type I noise...')
            stats = precompute_class_stats(X, y)

        if verbose:
            print(f'  [make_noise_cifar] Adding Type I noise: {n_type1} samples '
                  f'({type1_frac*100:.1f}% of original n={n})')

        ref_idx = rng.choice(n, n_type1, replace=True)
        X_t1 = np.empty((n_type1, X.shape[1]), dtype=X.dtype)
        y_t1 = y[ref_idx].copy()

        for i, c in enumerate(y_t1):
            st = stats[c]
            X_t1[i] = _generate_type1_sample(
                st['mu'], st['var'], st['tau'], rng, margin=type1_margin
            )

        X_parts.append(X_t1)
        y_parts.append(y_t1)


    # Label Outliers

    if n_type2 > 0:
        if verbose:
            print(f'  [make_noise_cifar] Adding Type II noise: {n_type2} samples '
                  f'({type2_frac*100:.1f}% of original n={n})')

        classes = np.unique(y)
        ref_idx = rng.choice(n, n_type2, replace=True)
        X_t2 = X[ref_idx].copy()
        y_t2 = y[ref_idx].copy()

        for i in range(n_type2):
            orig     = y_t2[i]
            other    = classes[classes != orig]
            y_t2[i]  = rng.choice(other)

        X_parts.append(X_t2)
        y_parts.append(y_t2)

    X_noisy = np.vstack(X_parts)
    y_noisy = np.concatenate(y_parts)

    if verbose and (n_type1 + n_type2) > 0:
        print(f'  [make_noise_cifar] Done. '
              f'Original={n}, +Type I={n_type1}, +Type II={n_type2} '
              f'→ total={len(y_noisy)}')

    return X_noisy, y_noisy



def verify_type1_distances(
    X_noisy: np.ndarray,
    y: np.ndarray,
    type1_indices: np.ndarray,
    stats: dict,
) -> dict:

    dists  = []
    taus   = []
    for i in type1_indices:
        c   = y[i]
        st  = stats[c]
        D   = np.sqrt(((X_noisy[i] - st['mu']) ** 2 / st['var']).sum())
        dists.append(D)
        taus.append(st['tau'])

    dists  = np.array(dists)
    taus   = np.array(taus)
    exceed = (dists > taus)

    return {
        'distances'      : dists,
        'thresholds'     : taus,
        'ratio_D_tau'    : dists / taus,
        'all_exceed_tau' : exceed.all(),
        'pct_exceed'     : exceed.mean() * 100,
    }


def _noise_npz_name(pca_components, subsample_train, type1_frac, type2_frac):

    t1  = int(round(type1_frac * 100))
    t2  = int(round(type2_frac * 100))
    sub = f'_sub{subsample_train}' if subsample_train else ''
    return f'cifar10_pca{pca_components}{sub}_t1_{t1:02d}_t2_{t2:02d}.npz'


def _get_data_dir():
    import os, sys
    _here   = os.path.dirname(os.path.abspath(__file__))
    _parent = os.path.dirname(_here) 
    for path in (_here, _parent):
        if path not in sys.path:
            sys.path.insert(0, path)
    from cifar10 import _DATA_DIR
    return _DATA_DIR


def prepare_noisy_datasets(
    pca_components: int = 256,
    noise_configs: list = None,
    subsample_train: int = None,
    random_state: int = 42,
    force: bool = False,
) -> dict:

    import os

    if noise_configs is None:
        grid = [0.0, 0.10, 0.20]
        noise_configs = [(t1, t2) for t1 in grid for t2 in grid]

    _DATA_DIR = _get_data_dir()
    from cifar10 import load_cifar10

    out_dir = os.path.join(_DATA_DIR, 'noisy')
    os.makedirs(out_dir, exist_ok=True)

    # 데이터 한 번만 로드
    print(f'Loading CIFAR-10 PCA-{pca_components}...')
    X_train, X_test, y_train, y_test = load_cifar10(
        pca_components=pca_components,
        subsample_train=subsample_train,
        random_state=random_state,
    )
    print(f'  X_train: {X_train.shape}')

    # 클래스 통계량도 한 번만 계산
    print('Computing class stats (Type I needs this)...')
    stats = precompute_class_stats(X_train, y_train)
    tau_summary = {c: f'{st["tau"]:.2f}' for c, st in stats.items()}
    print(f'  tau per class: {tau_summary}')

    paths = {}

    for (t1, t2) in noise_configs:
        name = _noise_npz_name(pca_components, subsample_train, t1, t2)
        path = os.path.join(out_dir, name)
        paths[(t1, t2)] = path

        if os.path.exists(path) and not force:
            print(f'  [skip] Already exists: {name}')
            continue

        label = f'Type I={t1*100:.0f}%  Type II={t2*100:.0f}%'
        print(f'  Generating {label}  → {name}')

        X_n, y_n = inject_noise_cifar(
            X_train, y_train,
            type1_frac=t1,
            type2_frac=t2,
            stats=stats,
            random_state=random_state,
            verbose=False,
        )
        np.savez_compressed(
            path,
            X_train=X_n, y_train=y_n,
            X_test=X_test,  y_test=y_test,
            type1_frac=np.array([t1]),
            type2_frac=np.array([t2]),
        )
        print(f'    Saved ({os.path.getsize(path) / 1e6:.1f} MB)')

    print(f'\nAll noisy datasets ready in: {out_dir}')
    return paths


def load_noisy_cifar10(
    type1_frac: float,
    type2_frac: float,
    pca_components: int = 256,
    subsample_train: int = None,
) -> tuple:

    import os
    _DATA_DIR = _get_data_dir()

    name = _noise_npz_name(pca_components, subsample_train, type1_frac, type2_frac)
    path = os.path.join(_DATA_DIR, 'noisy', name)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path}\n'
            f'→ Run first:  python datasets/make_noise_cifar.py --prepare'
        )

    data = np.load(path)
    return (
        data['X_train'].astype(np.float64),
        data['X_test'].astype(np.float64),
        data['y_train'],
        data['y_test'],
    )


if __name__ == '__main__':
    import sys, os, argparse

    _here   = os.path.dirname(os.path.abspath(__file__))
    _parent = os.path.dirname(_here)
    for _p in (_here, _parent):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from cifar10 import load_cifar10

    parser = argparse.ArgumentParser(description='CIFAR-10 noise injection tool')
    parser.add_argument('--prepare', action='store_true',
                        help='Pre-generate all noisy datasets and save to disk')
    parser.add_argument('--pca', type=int, default=256,
                        help='PCA components (default: 256)')
    parser.add_argument('--subsample', type=int, default=None,
                        help='Subsample train set (None = full 50k)')
    parser.add_argument('--grid', nargs='+', type=float,
                        default=[0.0, 0.10, 0.20],
                        help='Grid values for type1 and type2 fractions. '
                             'All combinations (t1, t2) will be generated. '
                             '(default: 0.0 0.10 0.20 → 3×3 = 9 files)')
    parser.add_argument('--force', action='store_true',
                        help='Re-generate even if files already exist')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.prepare:
        grid = args.grid
        configs = [(t1, t2) for t1 in grid for t2 in grid]
        print(f'Grid: type1 ∈ {[f"{g*100:.0f}%" for g in grid]}  '
              f'× type2 ∈ {[f"{g*100:.0f}%" for g in grid]}  '
              f'→ {len(configs)} files')

        prepare_noisy_datasets(
            pca_components=args.pca,
            noise_configs=configs,
            subsample_train=args.subsample,
            random_state=args.seed,
            force=args.force,
        )
    else:
        sub = args.subsample or 5000
        print(f'Smoke-test: subsample={sub}, pca={args.pca}')
        X_tr, X_te, y_tr, y_te = load_cifar10(pca_components=args.pca,
                                               subsample_train=sub,
                                               random_state=args.seed)
        print(f'  X_train: {X_tr.shape}')

        print('\nPrecomputing class stats...')
        stats = precompute_class_stats(X_tr, y_tr)
        print(f'  Class 0: tau={stats[0]["tau"]:.3f}  mean_D={stats[0]["D"].mean():.3f}')

        print('\nInjecting mixed noise (Type I=10%, Type II=5%)...')
        X_n, y_n = inject_noise_cifar(
            X_tr, y_tr,
            type1_frac=0.10, type2_frac=0.05,
            stats=stats, verbose=True,
        )

        rng     = np.random.default_rng(args.seed)
        n_type1 = int(len(y_tr) * 0.10)
        n_type2 = int(len(y_tr) * 0.05)
        all_idx   = rng.choice(len(y_tr), n_type1 + n_type2, replace=False)
        type1_idx = all_idx[:n_type1]

        report = verify_type1_distances(X_n, y_tr, type1_idx, stats)
        print(f'\n  Verification — Type I samples (n={len(type1_idx)}):')
        print(f'    D/tau ratio: min={report["ratio_D_tau"].min():.3f}  '
              f'mean={report["ratio_D_tau"].mean():.3f}  '
              f'max={report["ratio_D_tau"].max():.3f}')
        print(f'    All exceed tau: {report["all_exceed_tau"]}  '
              f'({report["pct_exceed"]:.1f}% exceed)')
        print('\nDone.')
