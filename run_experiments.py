"""
Multi-Dataset Model Comparison Experiments
Runs experiments on 4 datasets with Type I boundary noise and label noise
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datasets import inject_noise
from svm_models import NaiveSVM, ProbSVM, KNNSVM, SKiP
from multi_svm import OneVsRestSVM
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')


# Dataset configurations
datasets_config = {
    'breast_cancer': {
        'clean': 'datasets/breast_cancer/breast_cancer.npz',
        'noisy': {
            '5%': 'datasets/breast_cancer/fast_breast_cancer_type1_boundary_5pct.npz',
            '10%': 'datasets/breast_cancer/fast_breast_cancer_type1_boundary_10pct.npz',
            '15%': 'datasets/breast_cancer/fast_breast_cancer_type1_boundary_15pct.npz',
            '20%': 'datasets/breast_cancer/fast_breast_cancer_type1_boundary_20pct.npz',
        }
    },
    'iris_2feat': {
        'clean': 'datasets/iris_2feat/iris_2feat.npz',
        'noisy': {
            '5%': 'datasets/iris_2feat/fast_iris_2feat_type1_boundary_5pct.npz',
            '10%': 'datasets/iris_2feat/fast_iris_2feat_type1_boundary_10pct.npz',
            '15%': 'datasets/iris_2feat/fast_iris_2feat_type1_boundary_15pct.npz',
            '20%': 'datasets/iris_2feat/fast_iris_2feat_type1_boundary_20pct.npz',
        }
    },
    'titanic': {
        'clean': 'datasets/titanic/titanic.npz',
        'noisy': {
            '5%': 'datasets/titanic/fast_titanic_type1_boundary_5pct.npz',
            '10%': 'datasets/titanic/fast_titanic_type1_boundary_10pct.npz',
            '15%': 'datasets/titanic/fast_titanic_type1_boundary_15pct.npz',
            '20%': 'datasets/titanic/fast_titanic_type1_boundary_20pct.npz',
        }
    },
    'wine': {
        'clean': 'datasets/wine/wine.npz',
        'noisy': {
            '5%': 'datasets/wine/fast_wine_type1_boundary_5pct.npz',
            '10%': 'datasets/wine/fast_wine_type1_boundary_10pct.npz',
            '15%': 'datasets/wine/fast_wine_type1_boundary_15pct.npz',
            '20%': 'datasets/wine/fast_wine_type1_boundary_20pct.npz',
        }
    }
}

# Experiment configurations
C_values = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
k_values = [3, 5, 7]
feature_noise_levels = [None, '5%', '10%', '15%', '20%']  # None = clean data
label_noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]


def load_dataset(dataset_name, feature_noise_level=None, label_noise=0.0, random_state=42):
    """
    Load dataset from npz file and inject label noise.

    Parameters:
    - dataset_name: Name of the dataset ('breast_cancer', 'iris_2feat', 'titanic', 'wine')
    - feature_noise_level: None for clean data, or '5%', '10%', '15%', '20%' for pre-generated feature noise
    - label_noise: Proportion of labels to flip (0.0 to 1.0)
    - random_state: Random seed for reproducibility

    Returns:
    - X_train, X_test, y_train, y_test (all scaled)
    """
    if feature_noise_level is None:
        file_path = datasets_config[dataset_name]['clean']
    else:
        file_path = datasets_config[dataset_name]['noisy'][feature_noise_level]

    data = np.load(file_path)
    X = data['X_train']  # The full dataset is stored as X_train
    y = data['y_train']  # The full labels are stored as y_train

    # Inject label noise if needed
    if label_noise > 0:
        X, y = inject_noise(X, y, feature_noise=0.0, label_noise=label_noise, 
                           random_state=random_state, add_label_noise=False)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def run_single_experiment(args):
    """
    Run a single experiment with given configuration.
    Designed to be called by multiprocessing.

    Parameters:
    - args: Tuple of (dataset_name, feature_noise_level, label_noise, random_state)

    Returns:
    - List of result dictionaries for this configuration
    """
    dataset_name, feature_noise_level, label_noise, random_state = args

    # Load data with feature noise and inject label noise
    X_train, X_test, y_train, y_test = load_dataset(dataset_name, feature_noise_level, label_noise, random_state)

    results = []
    feature_noise_str = "Clean" if feature_noise_level is None else feature_noise_level
    label_noise_str = f"{int(label_noise * 100)}%"

    print(f"Starting: {dataset_name} | Feature: {feature_noise_str} | Label: {label_noise_str}")

    # Test each model with different C values
    for C in C_values:
        # NaiveSVM
        clf = OneVsRestSVM(NaiveSVM(C=C, kernel='linear', verbose=False))
        clf.fit(X_train, y_train)
        train_acc = (clf.predict(X_train) == y_train).mean()
        test_acc = (clf.predict(X_test) == y_test).mean()
        results.append({
            'Dataset': dataset_name,
            'Feature_Noise': feature_noise_str,
            'Label_Noise': label_noise_str,
            'Model': 'NaiveSVM',
            'C': C,
            'k': None,
            'Train Acc': train_acc,
            'Test Acc': test_acc
        })

        # ProbSVM
        clf = OneVsRestSVM(ProbSVM(C=C, kernel='linear', verbose=False))
        clf.fit(X_train, y_train)
        train_acc = (clf.predict(X_train) == y_train).mean()
        test_acc = (clf.predict(X_test) == y_test).mean()
        results.append({
            'Dataset': dataset_name,
            'Feature_Noise': feature_noise_str,
            'Label_Noise': label_noise_str,
            'Model': 'ProbSVM',
            'C': C,
            'k': None,
            'Train Acc': train_acc,
            'Test Acc': test_acc
        })

    # Test KNN-based models with different C and k values
    for C in C_values:
        for k in k_values:
            # KNNSVM
            clf = OneVsRestSVM(KNNSVM(C=C, k=k, kernel='linear', verbose=False))
            clf.fit(X_train, y_train)
            train_acc = (clf.predict(X_train) == y_train).mean()
            test_acc = (clf.predict(X_test) == y_test).mean()
            results.append({
                'Dataset': dataset_name,
                'Feature_Noise': feature_noise_str,
                'Label_Noise': label_noise_str,
                'Model': 'KNNSVM',
                'C': C,
                'k': k,
                'Train Acc': train_acc,
                'Test Acc': test_acc
            })

            # SKiP - multiply
            clf = OneVsRestSVM(SKiP(C=C, k=k, kernel='linear', verbose=False, combine_method='multiply'))
            clf.fit(X_train, y_train)
            train_acc = (clf.predict(X_train) == y_train).mean()
            test_acc = (clf.predict(X_test) == y_test).mean()
            results.append({
                'Dataset': dataset_name,
                'Feature_Noise': feature_noise_str,
                'Label_Noise': label_noise_str,
                'Model': 'SKiP-multiply',
                'C': C,
                'k': k,
                'Train Acc': train_acc,
                'Test Acc': test_acc
            })

            # SKiP - multiply (min-max scaling)
            clf = OneVsRestSVM(SKiP(C=C, k=k, kernel='linear', verbose=False, combine_method='multiply', scaling='minmax'))
            clf.fit(X_train, y_train)
            train_acc = (clf.predict(X_train) == y_train).mean()
            test_acc = (clf.predict(X_test) == y_test).mean()
            results.append({
                'Dataset': dataset_name,
                'Feature_Noise': feature_noise_str,
                'Label_Noise': label_noise_str,
                'Model': 'SKiP-multiply-minmax',
                'C': C,
                'k': k,
                'Train Acc': train_acc,
                'Test Acc': test_acc
            })

            # SKiP - average
            clf = OneVsRestSVM(SKiP(C=C, k=k, kernel='linear', verbose=False, combine_method='average'))
            clf.fit(X_train, y_train)
            train_acc = (clf.predict(X_train) == y_train).mean()
            test_acc = (clf.predict(X_test) == y_test).mean()
            results.append({
                'Dataset': dataset_name,
                'Feature_Noise': feature_noise_str,
                'Label_Noise': label_noise_str,
                'Model': 'SKiP-average',
                'C': C,
                'k': k,
                'Train Acc': train_acc,
                'Test Acc': test_acc
            })

            # SKiP - average (min-max scaling)
            clf = OneVsRestSVM(SKiP(C=C, k=k, kernel='linear', verbose=False, combine_method='average', scaling='minmax'))
            clf.fit(X_train, y_train)
            train_acc = (clf.predict(X_train) == y_train).mean()
            test_acc = (clf.predict(X_test) == y_test).mean()
            results.append({
                'Dataset': dataset_name,
                'Feature_Noise': feature_noise_str,
                'Label_Noise': label_noise_str,
                'Model': 'SKiP-average-minmax',
                'C': C,
                'k': k,
                'Train Acc': train_acc,
                'Test Acc': test_acc
            })

    print(f"Completed: {dataset_name} | Feature: {feature_noise_str} | Label: {label_noise_str} | Total results: {len(results)}")
    return results


def main():
    """Run all experiments in parallel and save results to CSV"""
    print("="*70)
    print("MULTI-DATASET MODEL COMPARISON EXPERIMENTS (MULTIPROCESSING)")
    print("="*70)
    print(f"\nDatasets: {list(datasets_config.keys())}")
    print(f"Feature noise levels: {['Clean' if x is None else x for x in feature_noise_levels]}")
    print(f"Label noise levels: {[f'{int(n*100)}%' for n in label_noise_levels]}")
    print(f"Models: NaiveSVM, ProbSVM, KNNSVM, SKiP (4 variants)")
    print(f"C values: {C_values}")
    print(f"k values: {k_values}")

    # Calculate total experiments
    num_base_models = len(C_values) * 2  # NaiveSVM, ProbSVM
    num_knn_models = len(C_values) * len(k_values) * 5  # KNNSVM + 4 SKiP variants
    total_per_noise_combo = num_base_models + num_knn_models
    total_experiments = total_per_noise_combo * len(feature_noise_levels) * len(label_noise_levels) * len(datasets_config)

    # Get number of CPUs
    n_cpus = cpu_count()
    print(f"\nTotal experiments: {total_experiments}")
    print(f"CPUs available: {n_cpus}")
    print(f"Using {n_cpus} parallel processes")
    print("\nStarting experiments...\n")

    # Prepare all experiment configurations
    experiment_configs = []
    for dataset_name in ['breast_cancer', 'iris_2feat', 'titanic', 'wine']:
        for feature_noise_level in feature_noise_levels:
            for label_noise in label_noise_levels:
                experiment_configs.append((dataset_name, feature_noise_level, label_noise, 42))

    print(f"Total configurations to process: {len(experiment_configs)}\n")

    # Run experiments in parallel
    with Pool(processes=n_cpus) as pool:
        all_results_list = pool.map(run_single_experiment, experiment_configs)

    # Flatten results list
    all_results = []
    for results in all_results_list:
        all_results.extend(results)

    # Convert to DataFrame
    df_all_results = pd.DataFrame(all_results)

    print(f"\n\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"{'='*70}")
    print(f"Total experiments: {len(df_all_results)}")
    print(f"Datasets: {df_all_results['Dataset'].nunique()}")
    print(f"Models: {df_all_results['Model'].nunique()}")
    print(f"Feature noise levels: {df_all_results['Feature_Noise'].nunique()}")
    print(f"Label noise levels: {df_all_results['Label_Noise'].nunique()}")

    # Save results to CSV
    output_filename = 'model_comparison_results.csv'
    df_all_results.to_csv(output_filename, index=False)
    print(f"\n✓ Results saved to {output_filename}")

    # Display sample results
    print(f"\nSample results:")
    print(df_all_results.head(10))

    # Display basic statistics
    print(f"\n\nDataset statistics:")
    print(df_all_results.groupby('Dataset').size())
    print(f"\nModel statistics:")
    print(df_all_results.groupby('Model').size())


if __name__ == '__main__':
    main()
