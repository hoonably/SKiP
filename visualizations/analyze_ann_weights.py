import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.svm_sgd import ANNSVM_SGD, SKiP_SGD

CIFAR10_NAMES = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
    4: 'deer',     5: 'dog',        6: 'frog', 7: 'horse',
    8: 'ship',     9: 'truck',
}


def compute_and_print(X, y, class_names, dataset_name, k, threshold, prob_pca=8):
    lines = []

    def emit(s=''):
        print(s)
        lines.append(s)

    emit(f"\n{'='*70}")
    emit(f"  {dataset_name}  (k={k}, threshold={threshold}, prob_pca={prob_pca})")
    emit(f"{'='*70}")
    emit(f"  Computing ANN + SKiP weights ({len(np.unique(y))} classes, n={len(y)})...")

    ann  = ANNSVM_SGD(k=k, random_state=42)
    w_i  = ann._compute_ann_weights(X, y)

    skip = SKiP_SGD(k=k, random_state=42, n_prob_components=prob_pca)
    p_i  = skip._compute_class_conditionals(X, y)
    s_i  = 0.5 * (w_i + p_i)

    def print_table(weights, weight_name):
        col      = f"{weight_name}<={threshold} (%)"
        mean_col = f"mean {weight_name}"
        header   = f"{'Class':<12} | {'N':>5} | {col:>14} | {mean_col:>10}"
        sep      = '-' * len(header)
        emit(f"\n  [ {weight_name} ]")
        emit(header)
        emit(sep)
        for d in np.unique(y):
            mask  = y == d
            below = (weights[mask] <= threshold).mean() * 100
            name  = class_names[int(d)] if class_names is not None else str(int(d))
            emit(f"{name:<12} | {mask.sum():>5} | {below:>13.1f}% | {weights[mask].mean():>10.3f}")
        emit(sep)
        all_below = (weights <= threshold).mean() * 100
        emit(f"{'ALL':<12} | {len(y):>5} | {all_below:>12.1f}% | {weights.mean():>10.3f}")

    print_table(w_i, "w_i")
    print_table(p_i, "p_i")
    print_table(s_i, "skip")

    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',   default='all',
                        choices=['all', 'iris', 'cifar10'],
                        help='Dataset to analyse (default: all)')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='w_i threshold (default: 0.01)')
    parser.add_argument('--k',         type=int,   default=10,
                        help='Number of ANN neighbors (default: 10)')
    parser.add_argument('--subsample', type=int,   default=50000,
                        help='Subsample size for CIFAR-10 (default: 50000)')
    parser.add_argument('--pca_cifar10', type=int, default=256,
                        help='PCA dims for CIFAR-10 (default: 256)')
    parser.add_argument('--prob_pca',   type=int, default=8,
                        help='PCA dims used for Mahalanobis (p_i) in SKiP weight (default: 8)')
    args = parser.parse_args()

    run = args.dataset
    all_lines = []

    # ---- Iris ----------------------------------------------------------------
    if run in ('all', 'iris'):
        from sklearn.datasets import load_iris
        iris = load_iris()
        all_lines += compute_and_print(
            X=iris.data, y=iris.target,
            class_names=list(iris.target_names),
            dataset_name='Iris',
            k=args.k, threshold=args.threshold,
            prob_pca=args.prob_pca,
        )

    # ---- CIFAR-10 ------------------------------------------------------------
    if run in ('all', 'cifar10'):
        from datasets.cifar10 import load_cifar10
        X, _, y, _ = load_cifar10(
            pca_components=args.pca_cifar10,
            subsample_train=args.subsample,
            random_state=42,
        )
        all_lines += compute_and_print(
            X=X, y=y,
            class_names=CIFAR10_NAMES,
            dataset_name=f'CIFAR-10 PCA-{args.pca_cifar10} (subsample={args.subsample})',
            k=args.k, threshold=args.threshold,
            prob_pca=args.prob_pca,
        )

    os.makedirs('./ann_weights', exist_ok=True)
    out_txt = f'./ann_weights/ann_weights_{run}_k{args.k}_thr{args.threshold}.txt'
    with open(out_txt, 'w') as f:
        f.write('\n'.join(all_lines))
    print(f"\nSaved: {out_txt}")


if __name__ == '__main__':
    main()
