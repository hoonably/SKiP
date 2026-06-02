import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import argparse
import os
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


def precompute_class_stats(X_train_scaled, y_train, percentile=99, categorical_mask=None):

    class_stats = {}
    classes = np.unique(y_train)

    if categorical_mask is not None:
        continuous_mask = ~categorical_mask
    else:
        continuous_mask = np.ones(X_train_scaled.shape[1], dtype=bool)

    for cls in classes:
        X_cls = X_train_scaled[y_train == cls]
        X_cls_continuous = X_cls[:, continuous_mask]

        mu = X_cls_continuous.mean(axis=0)
        cov = np.cov(X_cls_continuous.T) + np.eye(X_cls_continuous.shape[1]) * 1e-6
        L = np.linalg.cholesky(cov)
        inv_cov = np.linalg.inv(cov)
        diffs = X_cls_continuous - mu
        dists = np.einsum('ni,ij,nj->n', diffs, inv_cov, diffs)
        tau = np.percentile(dists, percentile)

        if categorical_mask is not None:
            mu_categorical = X_cls[:, categorical_mask].mean(axis=0)
        else:
            mu_categorical = None

        class_stats[cls] = {
            "mu": mu,
            "L": L,
            "tau": tau,
            "mu_categorical": mu_categorical,
        }

    return class_stats


def generate_type1_noise_with_boundary(
    X_train, y_train,
    ratios=[0.05, 0.10, 0.15, 0.20],
    random_state=42,
    output_dir=".",
    dataset_name="dataset",
    visualize=True,
    categorical_mask=None
):

    rng = default_rng(random_state)
    N = len(X_train)
    d = X_train.shape[1]

    if categorical_mask is not None:
        n_categorical = np.sum(categorical_mask)
        n_continuous = d - n_categorical
        print(f"Input data: N={N}, d={d} (continuous: {n_continuous}, categorical: {n_categorical})")
        print(f"Categorical features excluded from noise.")
    else:
        print(f"Input data: N={N}, d={d}")
    print(f"Target noise ratios: {[f'{r*100:.0f}%' for r in ratios]}")
    print("-" * 60)

    print("1. Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("2. Training base SVM...")
    base_svm = LinearSVC(C=1.0, max_iter=10000, random_state=random_state, dual='auto')
    base_svm.fit(X_train_scaled, y_train)
    train_acc = base_svm.score(X_train_scaled, y_train)
    print(f"   Base SVM trained (Train Accuracy: {train_acc:.4f})")

    print("3. Computing class statistics...")
    class_stats = precompute_class_stats(X_train_scaled, y_train, percentile=99, categorical_mask=categorical_mask)
    classes = np.array(list(class_stats.keys()))
    print(f"   Classes: {len(classes)}")

    print("\n4. Generating Type 1 noise (Cholesky + radial scaling)...")
    print("   Condition: Mahalanobis > 99th pct AND crosses decision boundary")
    print("-" * 60)

    crossed_outliers_X = []
    crossed_outliers_y = []
    crossed_predictions = []
    not_crossed_outliers_X = []
    not_crossed_outliers_y = []
    not_crossed_predictions = []

    max_ratio = max(ratios)
    max_outliers = int(N * max_ratio)
    attempt_count = 0
    saved_ratios = set()

    while len(crossed_outliers_X) < max_outliers:
        cls = rng.choice(classes)
        stats = class_stats[cls]
        mu = stats["mu"]
        L = stats["L"]
        tau = stats["tau"]
        mu_categorical = stats["mu_categorical"]

        if categorical_mask is not None:
            d_continuous = np.sum(~categorical_mask)
        else:
            d_continuous = d

        z = rng.standard_normal(size=d_continuous)
        mah_distance = np.dot(z, z)
        E = rng.exponential(scale=2 / tau)
        r2 = tau + E
        scale = np.sqrt(r2 / (mah_distance + 1e-12))
        z = z * scale
        x_candidate_continuous = mu + L @ z

        if categorical_mask is not None:
            x_candidate_scaled = np.zeros(d)
            x_candidate_scaled[~categorical_mask] = x_candidate_continuous
            x_candidate_scaled[categorical_mask] = np.round(mu_categorical)
        else:
            x_candidate_scaled = x_candidate_continuous

        attempt_count += 1
        pred_label = base_svm.predict(x_candidate_scaled.reshape(1, -1))[0]

        if pred_label != cls:
            crossed_outliers_X.append(x_candidate_scaled)
            crossed_outliers_y.append(cls)
            crossed_predictions.append(pred_label)
        else:
            not_crossed_outliers_X.append(x_candidate_scaled)
            not_crossed_outliers_y.append(cls)
            not_crossed_predictions.append(pred_label)

        current_count = len(crossed_outliers_X)

        if current_count % 100 == 0 or current_count in [int(N * r) for r in ratios]:
            accept_rate = current_count / attempt_count if attempt_count > 0 else 0
            total_generated = len(crossed_outliers_X) + len(not_crossed_outliers_X)
            print(f"   Boundary crossed: {current_count}/{max_outliers} "
                  f"(total generated: {total_generated}, accept rate: {accept_rate:.2%})")

        for ratio in ratios:
            target_count = int(N * ratio)
            if current_count >= target_count and ratio not in saved_ratios:
                save_dataset(
                    X_train, y_train,
                    crossed_outliers_X,
                    crossed_outliers_y,
                    scaler,
                    ratio,
                    output_dir,
                    dataset_name
                )
                saved_ratios.add(ratio)

    print("-" * 60)
    print(f"\nFinal stats:")
    print(f"  - Total attempts: {attempt_count}")
    print(f"  - Boundary crossed: {len(crossed_outliers_X)}")
    print(f"  - Not crossed: {len(not_crossed_outliers_X)}")
    print(f"  - Total generated: {len(crossed_outliers_X) + len(not_crossed_outliers_X)}")

    total_accepted = len(crossed_outliers_X) + len(not_crossed_outliers_X)
    if total_accepted > 0:
        cross_rate = len(crossed_outliers_X) / total_accepted
        print(f"  - Boundary cross rate: {cross_rate:.2%}")

    print(f"\nSaved ratios: {sorted([f'{r*100:.0f}%' for r in saved_ratios])}")

    if visualize:
        print("\n5. Generating visualizations...")
        vis_dir = Path(output_dir) / "fast_vis"
        vis_dir.mkdir(exist_ok=True)

        for idx, ratio in enumerate(ratios):
            target_count = int(N * ratio)
            all_outliers_X = crossed_outliers_X[:target_count] + not_crossed_outliers_X[:]
            all_outliers_y = crossed_outliers_y[:target_count] + not_crossed_outliers_y[:]
            all_predictions = crossed_predictions[:target_count] + not_crossed_predictions[:]

            visualize_noise_generation(
                X_train_scaled=X_train_scaled,
                y_train=y_train,
                base_svm=base_svm,
                scaler=scaler,
                accepted_outliers_X=all_outliers_X,
                accepted_outliers_y=all_outliers_y,
                accepted_predictions=all_predictions,
                ratio=ratio,
                vis_dir=vis_dir,
                dataset_name=dataset_name
            )

            visualize_per_class(
                X_train_scaled=X_train_scaled,
                y_train=y_train,
                base_svm=base_svm,
                scaler=scaler,
                accepted_outliers_X=all_outliers_X,
                accepted_outliers_y=all_outliers_y,
                accepted_predictions=all_predictions,
                ratio=ratio,
                vis_dir=vis_dir,
                dataset_name=dataset_name
            )
        print(f"   Visualizations saved: {vis_dir}")


def save_dataset(
    X_train_orig, y_train_orig,
    outliers_X_scaled, outliers_y,
    scaler,
    ratio,
    output_dir,
    dataset_name
):
    outliers_X_scaled_array = np.array(outliers_X_scaled)
    outliers_y_array = np.array(outliers_y)
    outliers_X_orig = scaler.inverse_transform(outliers_X_scaled_array)

    output_path = Path(output_dir) / f"fast_{dataset_name}_type1_boundary_{int(ratio*100)}pct.npz"
    np.savez(output_path, X_train=outliers_X_orig, y_train=outliers_y_array)

    print(f"\n✓ Saved: {output_path.name}")
    print(f"  - Original: {len(X_train_orig)}, Noise: {len(outliers_X_orig)}, "
          f"{ratio*100:.0f}% noise")


def find_dataset_files(base_dir="."):
    datasets = []
    for item in Path(base_dir).iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            npz_files = list(item.glob("*.npz"))
            original_files = [f for f in npz_files
                               if 'type1_boundary' not in f.name and not f.name.startswith('fast_')]
            if original_files:
                datasets.append((item.name, original_files[0]))
    return datasets


def visualize_noise_generation(
    X_train_scaled, y_train, base_svm, scaler,
    accepted_outliers_X, accepted_outliers_y, accepted_predictions,
    ratio, vis_dir, dataset_name
):
    pca = PCA(n_components=2, random_state=42)
    X_train_2d = pca.fit_transform(X_train_scaled)

    if len(accepted_outliers_X) > 0:
        outliers_2d = pca.transform(np.array(accepted_outliers_X))
    else:
        return

    classes = np.unique(y_train)
    n_classes = len(classes)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for idx, cls in enumerate(classes):
        mask = y_train == cls
        ax.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1],
                   c=[colors[idx]], label=f'Class {cls} (Original)',
                   alpha=0.6, s=50, edgecolors='k', linewidth=0.5)

    for idx, cls in enumerate(classes):
        mask = np.array(accepted_outliers_y) == cls
        if np.any(mask):
            ax.scatter(outliers_2d[mask, 0], outliers_2d[mask, 1],
                       c=[colors[idx]], marker='o', s=50,
                       label=f'Class {cls} (Noise)',
                       edgecolors='red', linewidth=1.5, alpha=0.9)

    plot_decision_boundary_pca(ax, base_svm, pca, X_train_scaled)
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title(f'{dataset_name} - Type 1 Noise ({ratio*100:.0f}%)\nOriginal Data + Noise Points',
                 fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1],
               c='lightgray', alpha=0.3, s=30, label='Original Data')

    outliers_y_arr = np.array(accepted_outliers_y)
    predictions_arr = np.array(accepted_predictions)

    crossed_mask = outliers_y_arr != predictions_arr
    if np.any(crossed_mask):
        ax.scatter(outliers_2d[crossed_mask, 0], outliers_2d[crossed_mask, 1],
                   c='red', marker='o', s=60,
                   label=f'Crossed Boundary ({np.sum(crossed_mask)})',
                   edgecolors='darkred', linewidth=1.5, alpha=0.9, zorder=5)

    not_crossed_mask = ~crossed_mask
    if np.any(not_crossed_mask):
        ax.scatter(outliers_2d[not_crossed_mask, 0], outliers_2d[not_crossed_mask, 1],
                   c='blue', marker='o', s=60,
                   label=f'Not Crossed ({np.sum(not_crossed_mask)})',
                   edgecolors='darkblue', linewidth=1.5, alpha=0.9, zorder=5)

    plot_decision_boundary_pca(ax, base_svm, pca, X_train_scaled, alpha=0.4)
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title(f'Noise Detail Analysis\nGenerated Noise: {len(accepted_outliers_X)} points',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    explained_var = pca.explained_variance_ratio_
    fig.suptitle(f'{dataset_name} - Type 1 Feature Noise Visualization ({ratio*100:.0f}%)\n'
                 f'PCA explained variance: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    output_path = vis_dir / f"{dataset_name}_type1_noise_{int(ratio*100)}pct.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ {output_path.name}")


def plot_decision_boundary_pca(ax, svm, pca, X_train_scaled, alpha=0.2):
    X_2d = pca.transform(X_train_scaled)
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    h = (x_max - x_min) / 200
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid_original = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
    Z = svm.predict(grid_original).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=alpha, cmap='RdYlBu')
    ax.contour(xx, yy, Z, colors='black', linewidths=1.5, alpha=0.5)


def visualize_per_class(
    X_train_scaled, y_train, base_svm, scaler,
    accepted_outliers_X, accepted_outliers_y, accepted_predictions,
    ratio, vis_dir, dataset_name
):
    if len(accepted_outliers_X) == 0:
        return

    pca = PCA(n_components=2, random_state=42)
    X_train_2d = pca.fit_transform(X_train_scaled)
    outliers_2d = pca.transform(np.array(accepted_outliers_X))

    classes = np.unique(y_train)
    outliers_y_arr = np.array(accepted_outliers_y)
    predictions_arr = np.array(accepted_predictions)

    for target_cls in classes:
        class_dir = vis_dir / f"class{target_cls}"
        class_dir.mkdir(exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        other_original_mask = y_train != target_cls
        ax.scatter(X_train_2d[other_original_mask, 0], X_train_2d[other_original_mask, 1],
                   c='lightgray', alpha=0.3, s=30, label='Other Classes (Original)')

        target_original_mask = y_train == target_cls
        ax.scatter(X_train_2d[target_original_mask, 0], X_train_2d[target_original_mask, 1],
                   c='lightcoral', alpha=0.8, s=50,
                   edgecolors='salmon', linewidth=1,
                   label=f'Class {target_cls} (Original)')

        other_mask = outliers_y_arr != target_cls
        if np.any(other_mask):
            ax.scatter(outliers_2d[other_mask, 0], outliers_2d[other_mask, 1],
                       c='gray', marker='o', s=50,
                       alpha=0.3, edgecolors='darkgray', linewidth=1)

        target_mask = outliers_y_arr == target_cls
        if np.any(target_mask):
            target_predictions = predictions_arr[target_mask]
            target_outliers_2d = outliers_2d[target_mask]

            crossed_in_target = target_predictions != target_cls
            if np.any(crossed_in_target):
                ax.scatter(target_outliers_2d[crossed_in_target, 0],
                           target_outliers_2d[crossed_in_target, 1],
                           c='red', marker='o', s=80,
                           label=f'Class {target_cls} - Crossed ({np.sum(crossed_in_target)})',
                           edgecolors='darkred', linewidth=2, alpha=0.95, zorder=5)

            not_crossed_in_target = target_predictions == target_cls
            if np.any(not_crossed_in_target):
                ax.scatter(target_outliers_2d[not_crossed_in_target, 0],
                           target_outliers_2d[not_crossed_in_target, 1],
                           c='orange', marker='o', s=80,
                           label=f'Class {target_cls} - Not Crossed ({np.sum(not_crossed_in_target)})',
                           edgecolors='darkorange', linewidth=2, alpha=0.95, zorder=5)

        plot_decision_boundary_pca(ax, base_svm, pca, X_train_scaled, alpha=0.3)
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_title(f'{dataset_name} - Class {target_cls} Focus ({ratio*100:.0f}%)\n'
                     f'Red: Class {target_cls} noise, Gray: Other classes',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = class_dir / f"{dataset_name}_class{target_cls}_{int(ratio*100)}pct.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"   ✓ Class-specific visualizations ({len(classes)} classes)")


def main():
    parser = argparse.ArgumentParser(
        description="Type 1 Feature Noise Generator with Decision Boundary Constraint (Fast Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python make_noise_fast.py
    python make_noise_fast.py --seed 42 --ratios 0.05 0.10 0.15 0.20
        """
    )
    parser.add_argument('--ratios', type=float, nargs='+',
                        default=[0.05, 0.10, 0.15, 0.20],
                        help='Noise ratio list (default: 0.05 0.10 0.15 0.20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-vis', action='store_true',
                        help='Skip visualization generation')

    args = parser.parse_args()

    print("=" * 60)
    print("Type 1 Feature Noise Generator (Fast Version)")
    print("=" * 60)
    print(f"Config: seed={args.seed}, ratios={args.ratios}\n")

    datasets = find_dataset_files(".")

    if not datasets:
        print("Error: no datasets found.")
        print("Make sure .npz files exist in subdirectories of the current directory.")
        return

    print(f"Found datasets: {len(datasets)}\n")

    for idx, (dataset_name, npz_path) in enumerate(datasets, 1):
        print("=" * 60)
        print(f"[{idx}/{len(datasets)}] Dataset: {dataset_name}")
        print("=" * 60)
        print(f"Input: {npz_path}")

        try:
            data = np.load(npz_path)
            X_train = data['X_train']
            y_train = data['y_train']
            print(f"Loaded: X_train={X_train.shape}, y_train={y_train.shape}")

            output_dir = npz_path.parent

            categorical_mask = None
            if dataset_name == "titanic":
                # last 3 features are one-hot encoded: Sex_male, Embarked_Q, Embarked_S
                categorical_mask = np.zeros(X_train.shape[1], dtype=bool)
                categorical_mask[-3:] = True
                print(f"Titanic: categorical features (indices {np.where(categorical_mask)[0].tolist()}) excluded from noise.")

            generate_type1_noise_with_boundary(
                X_train=X_train,
                y_train=y_train,
                ratios=args.ratios,
                random_state=args.seed,
                output_dir=output_dir,
                dataset_name=dataset_name,
                visualize=not args.no_vis,
                categorical_mask=categorical_mask
            )

            print(f"\n✓ {dataset_name} done!\n")

        except Exception as e:
            print(f"\n✗ Error processing {dataset_name}: {e}\n")
            continue

    print("=" * 60)
    print("All datasets processed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
