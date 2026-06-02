import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.svm_sgd import ANNSVM_SGD, ProbSVM_SGD

CIFAR10_NAMES = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
    4: 'deer',     5: 'dog',        6: 'frog', 7: 'horse',
    8: 'ship',     9: 'truck',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',   default='cifar10', choices=['cifar10', 'iris'])
    p.add_argument('--classes',   nargs=2, type=int, default=None,
                   help='Two class indices to compare. '
                        'Defaults: cifar10→3,5 (cat vs dog)  iris→0,1 (setosa vs versicolor)')
    p.add_argument('--pca',       type=int, default=None,
                   help='PCA dims (default: 256 for cifar10)')
    p.add_argument('--subsample', type=int, default=50000)
    p.add_argument('--k',         type=int, default=10)
    p.add_argument('--n_prob',    type=int, default=8)
    p.add_argument('--threshold', type=float, default=0.3)
    p.add_argument('--seed',      type=int,   default=42)
    return p.parse_args()


def load_data(args):
    if args.dataset == 'iris':
        from sklearn.datasets import load_iris
        iris = load_iris()
        classes = args.classes or [0, 1]
        print("Loading Iris dataset...")
        X_all, y_all = iris.data, iris.target
        iris_names = list(iris.target_names)
        class_label = lambda c: iris_names[c]
    else:
        from datasets.cifar10 import load_cifar10
        pca = args.pca or 256
        classes = args.classes or [3, 5]
        print(f"Loading CIFAR-10 PCA-{pca} (all training data)...")
        X_all, _, y_all, _ = load_cifar10(
            pca_components=pca,
            random_state=args.seed,
        )
        class_label = lambda c: CIFAR10_NAMES[c]

    mask = np.isin(y_all, classes)
    X, y = X_all[mask], y_all[mask]
    print(f"Binary subset: {len(y)} samples  "
          f"({class_label(classes[0])}: {(y==classes[0]).sum()}, "
          f"{class_label(classes[1])}: {(y==classes[1]).sum()})")
    return X, y, classes, class_label


def main():
    args = parse_args()

    X, y, classes, class_label = load_data(args)

    # ---- Compute weights --------------------------------------------------
    print(f"\nComputing ANNSVM weights (k={args.k})...")
    ann = ANNSVM_SGD(k=args.k, epochs=1, random_state=args.seed)
    ann.fit(X, y)
    n_i = ann.w_i_

    print(f"Computing ProbSVM weights (n_prob={args.n_prob})...")
    prob = ProbSVM_SGD(epochs=1, n_prob_components=args.n_prob, random_state=args.seed)
    prob.fit(X, y)
    p_i = prob.p_i_

    # ---- Stats ------------------------------------------------------------
    thr = args.threshold
    boundary_mask = n_i < thr
    print(f"\n{'─'*52}")
    print(f"ANNSVM  n_i : mean={n_i.mean():.3f}  std={n_i.std():.3f}  "
          f"<{thr}: {boundary_mask.mean()*100:.1f}%")
    print(f"ProbSVM p_i : mean={p_i.mean():.3f}  std={p_i.std():.3f}  "
          f"<{thr}: {(p_i<thr).mean()*100:.1f}%")
    print(f"{'─'*52}")
    for c in classes:
        cm = y == c
        print(f"  {class_label(c)}: boundary {boundary_mask[cm].mean()*100:.1f}% "
              f"| n_i mean={n_i[cm].mean():.3f}  p_i mean={p_i[cm].mean():.3f}")

    # ---- 2D projection (top 2 PCA components) ----------------------------
    X2 = X[:, :2]

    def tight_ticks(ax):
        xlo, xhi = X2[:, 0].min(), X2[:, 0].max()
        ylo, yhi = X2[:, 1].min(), X2[:, 1].max()
        xticks = AutoLocator().tick_values(xlo, xhi)
        yticks = AutoLocator().tick_values(ylo, yhi)
        xticks = xticks[(xticks >= xlo) & (xticks <= xhi)]
        yticks = yticks[(yticks >= ylo) & (yticks <= yhi)]
        if len(xticks) >= 2:
            ax.set_xlim(xticks[0], xticks[-1])
            ax.set_xticks(xticks)
        if len(yticks) >= 2:
            ax.set_ylim(yticks[0], yticks[-1])
            ax.set_yticks(yticks)

    s     = 7
    alpha = 0.55
    c0, c1 = '#2196F3', '#F44336'

    lbl0, lbl1 = class_label(classes[0]), class_label(classes[1])
    tag = f"{args.dataset}_{lbl0}_vs_{lbl1}".replace(' ', '_')

    os.makedirs('./ann_weights', exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(X2[y==classes[0], 0], X2[y==classes[0], 1],
               c=c0, alpha=alpha, s=s, label=lbl0)
    ax.scatter(X2[y==classes[1], 0], X2[y==classes[1], 1],
               c=c1, alpha=alpha, s=s, label=lbl1)
    ax.scatter(X2[boundary_mask, 0], X2[boundary_mask, 1],
               c='yellow', edgecolors='black', linewidths=0.3,
               alpha=0.9, s=s * 3,
               label=f'$n_i < {thr}$')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(True, alpha=0.2)
    tight_ticks(ax)

    os.makedirs('./ann_weights/pdf', exist_ok=True)
    out_png = f'./ann_weights/{tag}_samples_with_ann_weights.png'
    out_pdf = f'./ann_weights/pdf/{tag}_samples_with_ann_weights.pdf'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == '__main__':
    main()
