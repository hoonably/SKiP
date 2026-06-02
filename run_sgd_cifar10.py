import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.cifar10          import load_cifar10
from datasets.make_noise_cifar import (inject_noise_cifar, precompute_class_stats,
                                        prepare_noisy_datasets, load_noisy_cifar10)
from models.svm_sgd   import NaiveSVM_SGD, ProbSVM_SGD, ANNSVM_SGD, SKiP_SGD
from models.multi_svm import OneVsRestSVM

os.makedirs('results', exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description='SGD-SVM CIFAR-10 experiment')
    p.add_argument('--pca',        type=int,   default=256,
                   help='PCA components to use (default: 256)')
    p.add_argument('--subsample',  type=int,   default=10000,
                   help='Subsample training set to N samples (None = full 50k)')
    p.add_argument('--C',          type=float, default=1,
                   help='SVM regularisation C (default: 1.0)')
    p.add_argument('--epochs',     type=int,   default=20,
                   help='SGD epochs (default: 20)')
    p.add_argument('--batch_size', type=int,   default=256,
                   help='Mini-batch size (default: 256)')
    p.add_argument('--k',          type=int,   default=10,
                   help='ANN neighbours for ANNSVM/SKiP (default: 10)')
    p.add_argument('--lr',         type=float, default=0.01,
                   help='Initial learning rate (default: 0.01)')
    p.add_argument('--rbf',        action='store_true',
                   help='Also run RBF Kernelized Pegasos (slow, budget-limited)')
    p.add_argument('--budget',     type=int,   default=500,
                   help='Kernelized Pegasos SV budget for RBF (default: 500)')
    p.add_argument('--noise_grid', nargs='+', type=float,
                   default=[0.0, 0.10, 0.20],
                   help='Grid values for feature outliers/label outliers fractions. '
                        'All (t1, t2) combinations are tested. '
                        '(default: 0.0 0.20 0.40 0.60 → 4×4 = 16 conditions)')
    p.add_argument('--prepare_noise', action='store_true',
                   help='Pre-generate noisy datasets to disk before experiment '
                        '(recommended for full 50k run; skips already-saved files)')
    p.add_argument('--n_prob',     type=int,   default=32,
                   help='# PCA components used for ProbSVM/SKiP Gaussian weight '
                        'computation (default: 32). Prevents weight collapse in '
                        'high-d. Set to 0 to use all PCA components (risky).')
    p.add_argument('--n_ann',      type=int,   default=0,
                   help='# PCA components for ANNSVM/SKiP ANN distance computation '
                        '(default: 32). 0 = use all (curse of dimensionality).')
    p.add_argument('--seed',       type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_sgd_models(args, kernel='linear'):
    """Return dict {name: OneVsRestSVM instance}."""
    n_prob = args.n_prob if args.n_prob > 0 else None
    n_ann  = args.n_ann  if args.n_ann  > 0 else None

    base = dict(
        C=args.C, kernel=kernel, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr,
        lr_schedule='pegasos', class_weight='balanced',
        verbose=False, random_state=args.seed,
    )
    rbf_extra = dict(budget=args.budget)

    models = {
        'NaiveSVM_SGD': OneVsRestSVM(NaiveSVM_SGD(**base)),
        'ProbSVM_SGD' : OneVsRestSVM(ProbSVM_SGD( **base, n_prob_components=n_prob)),
        'ANNSVM_SGD'  : OneVsRestSVM(ANNSVM_SGD(  **base, k=args.k,
                                                n_ann_components=n_ann)),
        'SKiP_SGD'   : OneVsRestSVM(SKiP_SGD(    **base, k=args.k,
                                                combine_method='average',
                                                n_prob_components=n_prob,
                                                n_ann_components=n_ann)),
    }
    if args.rbf and kernel == 'rbf':
        models = {
            'NaiveSVM_SGD_rbf': OneVsRestSVM(NaiveSVM_SGD(**base, **rbf_extra)),
            'SKiP_SGD_rbf'    : OneVsRestSVM(SKiP_SGD(    **base, **rbf_extra, k=args.k,
                                                         n_ann_components=n_ann)),
        }
    return models


def _load_or_generate(t1, t2, X_train, y_train, stats, args):
    """Load noisy dataset from cache or generate on-the-fly; return clean data as-is for (0,0)."""
    if t1 == 0.0 and t2 == 0.0:
        return X_train, y_train

    try:
        X_n, _, y_n, _ = load_noisy_cifar10(
            t1, t2,
            pca_components=args.pca,
            subsample_train=args.subsample,
        )
        return X_n, y_n
    except FileNotFoundError:
        X_n, y_n = inject_noise_cifar(
            X_train, y_train,
            type1_frac=t1, type2_frac=t2,
            stats=stats, random_state=args.seed, verbose=False,
        )
        return X_n, y_n


def experiment_noise(X_train, X_test, y_train, y_test, args):
    print('\n' + '='*68)
    print('Noise Robustness  (Feature Outliers × Label Outliers grid)')
    print('='*68)

    grid        = sorted(args.noise_grid)
    all_configs = [(t1, t2) for t1 in grid for t2 in grid]

    print('  Precomputing class stats...')
    stats = precompute_class_stats(X_train, y_train)

    models_lin  = build_sgd_models(args, kernel='linear')
    model_names = list(models_lin.keys())
    records     = []

    header_models = '  '.join(f'{n[:11]:>11}' for n in model_names)
    print(f"\n  {'Feature(%)':>10} {'Label(%)':>8}  {header_models}")
    print('  ' + '-'*(10 + 1 + 8 + 2 + len(header_models)))

    for (t1, t2) in all_configs:
        X_n, y_n = _load_or_generate(t1, t2, X_train, y_train, stats, args)

        row  = {'feat_frac': t1, 'label_frac': t2}
        accs = []
        for name in model_names:
            clf_fresh = build_sgd_models(args, kernel='linear')[name]
            clf_fresh.fit(X_n, y_n)
            acc = (clf_fresh.predict(X_test) == y_test).mean()
            row[name] = acc
            accs.append(f'{acc:>11.3f}')

        records.append(row)
        tag = '(clean)' if t1 == 0 and t2 == 0 else ''
        print(f"  {t1*100:>10.0f} {t2*100:>8.0f}  " + '  '.join(accs) + f'  {tag}')

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print('\n' + '='*68)
    print('  SGD-based SVM on CIFAR-10')
    print('='*68)
    print(f'  PCA components  : {args.pca}')
    print(f'  Subsample train : {args.subsample if args.subsample else "full 50k"}')
    print(f'  n_prob_comp     : {args.n_prob} (Gaussian weight dim for ProbSVM/SKiP)')
    print(f'  n_ann_comp      : {args.n_ann} (ANN distance dim for ANNSVM/SKiP)')
    print(f'  C={args.C}  epochs={args.epochs}  batch={args.batch_size}  '
          f'lr={args.lr}  k={args.k}')

    print('\nLoading CIFAR-10...')
    X_train, X_test, y_train, y_test = load_cifar10(
        pca_components=args.pca,
        subsample_train=args.subsample,
        random_state=args.seed,
    )
    print(f'  X_train: {X_train.shape}  y_train: {y_train.shape}')
    print(f'  X_test:  {X_test.shape}   y_test:  {y_test.shape}')
    print(f'  Classes: {np.unique(y_train).tolist()}')

    if args.prepare_noise:
        grid = args.noise_grid
        noise_configs = [(t1, t2) for t1 in grid for t2 in grid
                         if not (t1 == 0.0 and t2 == 0.0)]
        print(f'\nPre-generating noisy datasets '
              f'({len(noise_configs)} combos from {len(grid)}×{len(grid)} grid)...')
        prepare_noisy_datasets(
            pca_components=args.pca,
            noise_configs=noise_configs,
            subsample_train=args.subsample,
            random_state=args.seed,
        )

    df_noise = experiment_noise(X_train, X_test, y_train, y_test, args)

    df_noise.to_csv('results/cifar10_sgd_results.csv', index=False)
    print('\n  Results saved to results/cifar10_sgd_results.csv')

    print('\n' + '='*68)
    print('  Done.')
    print('='*68)


if __name__ == '__main__':
    main()
