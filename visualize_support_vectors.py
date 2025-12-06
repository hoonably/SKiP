"""
Support Vector Count Comparison Experiment
Small-scale experiment to compare number of support vectors between models
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datasets import inject_noise
from models.svm_models import NaiveSVM, ProbSVM, KNNSVM, SKiP
from models.multi_svm import OneVsRestSVM
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
    'breast_cancer_pca': {
        'clean': 'datasets/breast_cancer_pca/breast_cancer_pca.npz',
        'noisy': {
            '5%': 'datasets/breast_cancer_pca/fast_breast_cancer_pca_type1_boundary_5pct.npz',
            '10%': 'datasets/breast_cancer_pca/fast_breast_cancer_pca_type1_boundary_10pct.npz',
            '15%': 'datasets/breast_cancer_pca/fast_breast_cancer_pca_type1_boundary_15pct.npz',
            '20%': 'datasets/breast_cancer_pca/fast_breast_cancer_pca_type1_boundary_20pct.npz',
        }
    },
    'iris': {
        'clean': 'datasets/iris/iris.npz',
        'noisy': {
            '5%': 'datasets/iris/fast_iris_type1_boundary_5pct.npz',
            '10%': 'datasets/iris/fast_iris_type1_boundary_10pct.npz',
            '15%': 'datasets/iris/fast_iris_type1_boundary_15pct.npz',
            '20%': 'datasets/iris/fast_iris_type1_boundary_20pct.npz',
        }
    },
    'iris_pca': {
        'clean': 'datasets/iris_pca/iris_pca.npz',
        'noisy': {
            '5%': 'datasets/iris_pca/fast_iris_pca_type1_boundary_5pct.npz',
            '10%': 'datasets/iris_pca/fast_iris_pca_type1_boundary_10pct.npz',
            '15%': 'datasets/iris_pca/fast_iris_pca_type1_boundary_15pct.npz',
            '20%': 'datasets/iris_pca/fast_iris_pca_type1_boundary_20pct.npz',
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
    'titanic_pca': {
        'clean': 'datasets/titanic_pca/titanic_pca.npz',
        'noisy': {
            '5%': 'datasets/titanic_pca/fast_titanic_pca_type1_boundary_5pct.npz',
            '10%': 'datasets/titanic_pca/fast_titanic_pca_type1_boundary_10pct.npz',
            '15%': 'datasets/titanic_pca/fast_titanic_pca_type1_boundary_15pct.npz',
            '20%': 'datasets/titanic_pca/fast_titanic_pca_type1_boundary_20pct.npz',
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
    },
    'wine_pca': {
        'clean': 'datasets/wine_pca/wine_pca.npz',
        'noisy': {
            '5%': 'datasets/wine_pca/fast_wine_pca_type1_boundary_5pct.npz',
            '10%': 'datasets/wine_pca/fast_wine_pca_type1_boundary_10pct.npz',
            '15%': 'datasets/wine_pca/fast_wine_pca_type1_boundary_15pct.npz',
            '20%': 'datasets/wine_pca/fast_wine_pca_type1_boundary_20pct.npz',
        }
    }
}


def load_dataset(dataset_name, feature_noise_level=None, label_noise=0.0, random_state=42):
    """
    Load dataset from npz file and inject label noise.
    
    Parameters:
    - dataset_name: Name of the dataset
    - feature_noise_level: None for clean data, or '5%', '10%', '15%', '20%'
    - label_noise: Proportion of labels to flip (0.0 to 1.0)
    - random_state: Random seed for reproducibility
    
    Returns:
    - X_train, X_test, y_train, y_test (all scaled)
    """
    # Load clean data
    clean_file_path = datasets_config[dataset_name]['clean']
    clean_data = np.load(clean_file_path)
    X_clean = clean_data['X_train']
    y_clean = clean_data['y_train']

    # Inject label noise to clean data if needed
    if label_noise > 0:
        X_clean, y_clean = inject_noise(X_clean, y_clean, feature_noise=0.0, label_noise=label_noise, 
                                        random_state=random_state, add_label_noise=False)

    # If feature noise is specified, load and concatenate feature noise data
    if feature_noise_level is not None:
        noise_file_path = datasets_config[dataset_name]['noisy'][feature_noise_level]
        noise_data = np.load(noise_file_path)
        X_noise = noise_data['X_train']
        y_noise = noise_data['y_train']
        
        # Concatenate clean data (with label noise) and feature noise data
        X = np.vstack([X_clean, X_noise])
        y = np.concatenate([y_clean, y_noise])
    else:
        X = X_clean
        y = y_clean

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def count_support_vectors(clf):
    """
    Count total number of support vectors across all binary classifiers in OneVsRestSVM
    
    Parameters:
    - clf: Trained OneVsRestSVM model
    
    Returns:
    - Total number of support vectors
    """
    total_sv = 0
    for estimator in clf.estimators_:
        total_sv += len(estimator.support_)
    return total_sv


def run_experiment():
    """
    Run support vector count comparison experiment
    """
    print("="*70)
    print("SUPPORT VECTOR COUNT COMPARISON EXPERIMENT")
    print("="*70)
    
    # Small-scale configuration
    datasets = ['breast_cancer', 'breast_cancer_pca', 'iris', 'iris_pca', 
                'titanic', 'titanic_pca', 'wine', 'wine_pca']
    noise_configs = [
        (None, 0.0),    # (0%, 0%)
        ('5%', 0.05),   # (5%, 5%)
        ('10%', 0.10),  # (10%, 10%)
        ('15%', 0.15),  # (15%, 15%)
        ('20%', 0.20)   # (20%, 20%)
    ]
    
    C = 1.0  # Fixed C value for simplicity
    k = 5    # Fixed k value for SKiP
    kernel = 'rbf'  # Fixed kernel
    random_state = 42
    
    print(f"\nDatasets: {datasets}")
    print(f"Noise configurations (Feature%, Label%): {[(f if f else '0%', f'{int(l*100)}%') for f, l in noise_configs]}")
    print(f"Models: NaiveSVM, ProbSVM, KNNSVM, SKiP-average")
    print(f"Fixed parameters: C={C}, k={k}, kernel={kernel}\n")
    
    results = []
    total_experiments = len(datasets) * len(noise_configs) * 4  # 4 models
    current = 0
    
    for dataset_name in datasets:
        for feature_noise, label_noise in noise_configs:
            feature_str = "0%" if feature_noise is None else feature_noise
            label_str = f"{int(label_noise * 100)}%"
            
            print(f"[{current+1}/{total_experiments}] Processing {dataset_name} | Feature: {feature_str} | Label: {label_str}")
            
            # Load data
            X_train, X_test, y_train, y_test = load_dataset(
                dataset_name, feature_noise, label_noise, random_state
            )
            
            # Train NaiveSVM
            clf_naive = OneVsRestSVM(NaiveSVM(C=C, kernel=kernel, verbose=False))
            clf_naive.fit(X_train, y_train)
            sv_naive = count_support_vectors(clf_naive)
            train_acc_naive = (clf_naive.predict(X_train) == y_train).mean()
            test_acc_naive = (clf_naive.predict(X_test) == y_test).mean()
            
            results.append({
                'Dataset': dataset_name,
                'Feature_Noise': feature_str,
                'Label_Noise': label_str,
                'Model': 'NaiveSVM',
                'Support_Vectors': sv_naive,
                'Train_Acc': train_acc_naive,
                'Test_Acc': test_acc_naive,
                'Train_Size': len(y_train)
            })
            current += 1
            
            # Train ProbSVM
            clf_prob = OneVsRestSVM(ProbSVM(C=C, kernel=kernel, verbose=False))
            clf_prob.fit(X_train, y_train)
            sv_prob = count_support_vectors(clf_prob)
            train_acc_prob = (clf_prob.predict(X_train) == y_train).mean()
            test_acc_prob = (clf_prob.predict(X_test) == y_test).mean()
            
            results.append({
                'Dataset': dataset_name,
                'Feature_Noise': feature_str,
                'Label_Noise': label_str,
                'Model': 'ProbSVM',
                'Support_Vectors': sv_prob,
                'Train_Acc': train_acc_prob,
                'Test_Acc': test_acc_prob,
                'Train_Size': len(y_train)
            })
            current += 1
            
            # Train KNNSVM
            clf_knn = OneVsRestSVM(KNNSVM(C=C, k=k, kernel=kernel, verbose=False))
            clf_knn.fit(X_train, y_train)
            sv_knn = count_support_vectors(clf_knn)
            train_acc_knn = (clf_knn.predict(X_train) == y_train).mean()
            test_acc_knn = (clf_knn.predict(X_test) == y_test).mean()
            
            results.append({
                'Dataset': dataset_name,
                'Feature_Noise': feature_str,
                'Label_Noise': label_str,
                'Model': 'KNNSVM',
                'Support_Vectors': sv_knn,
                'Train_Acc': train_acc_knn,
                'Test_Acc': test_acc_knn,
                'Train_Size': len(y_train)
            })
            current += 1
            
            # Train SKiP-average
            clf_skip = OneVsRestSVM(SKiP(C=C, k=k, kernel=kernel, verbose=False, combine_method='average'))
            clf_skip.fit(X_train, y_train)
            sv_skip = count_support_vectors(clf_skip)
            train_acc_skip = (clf_skip.predict(X_train) == y_train).mean()
            test_acc_skip = (clf_skip.predict(X_test) == y_test).mean()
            
            results.append({
                'Dataset': dataset_name,
                'Feature_Noise': feature_str,
                'Label_Noise': label_str,
                'Model': 'SKiP-average',
                'Support_Vectors': sv_skip,
                'Train_Acc': train_acc_skip,
                'Test_Acc': test_acc_skip,
                'Train_Size': len(y_train)
            })
            current += 1
            
            print(f"  NaiveSVM: {sv_naive} | ProbSVM: {sv_prob} | KNNSVM: {sv_knn} | SKiP: {sv_skip} SVs")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Add SV ratio column
    df['SV_Ratio'] = df['Support_Vectors'] / df['Train_Size']
    
    # Save results
    df.to_csv('support_vector_comparison.csv', index=False)
    print(f"\n✓ Results saved to support_vector_comparison.csv")
    
    return df


def visualize_results(df):
    """
    Create visualizations for support vector count comparison
    
    Parameters:
    - df: DataFrame with experiment results
    """
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Create directories if they don't exist
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('visualizations/support_vector', exist_ok=True)
    os.makedirs('visualizations/support_vector/pdf', exist_ok=True)
    
    # Set style
    plt.rcParams['figure.figsize'] = (16, 12)
    
    # Create a combined noise label
    df['Noise_Level'] = df['Feature_Noise'] + '/' + df['Label_Noise']
    
    # Filter only breast_cancer and titanic datasets
    selected_datasets = ['iris_pca', 'titanic_pca']
    
    # Create subplots: 1 row x 2 columns for 2 datasets
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    for idx, dataset in enumerate(selected_datasets):
        df_dataset = df[df['Dataset'] == dataset]
        
        # Prepare data for plotting
        noise_levels = ['0%/0%', '5%/5%', '10%/10%', '15%/15%', '20%/20%']
        
        sv_naive = []
        sv_prob = []
        sv_knn = []
        sv_skip = []
        
        for noise in noise_levels:
            naive_val = df_dataset[(df_dataset['Model'] == 'NaiveSVM') & 
                                   (df_dataset['Noise_Level'] == noise)]['Support_Vectors'].values[0]
            prob_val = df_dataset[(df_dataset['Model'] == 'ProbSVM') & 
                                  (df_dataset['Noise_Level'] == noise)]['Support_Vectors'].values[0]
            knn_val = df_dataset[(df_dataset['Model'] == 'KNNSVM') & 
                                 (df_dataset['Noise_Level'] == noise)]['Support_Vectors'].values[0]
            skip_val = df_dataset[(df_dataset['Model'] == 'SKiP-average') & 
                                  (df_dataset['Noise_Level'] == noise)]['Support_Vectors'].values[0]
            sv_naive.append(naive_val)
            sv_prob.append(prob_val)
            sv_knn.append(knn_val)
            sv_skip.append(skip_val)
        
        # Plot
        x = np.arange(len(noise_levels))
        width = 0.23
        
        # Use Tableau 10 colors
        tab10_colors = plt.cm.tab10.colors
        
        bars1 = axes[idx].bar(x - 1.5*width, sv_naive, width, label='NaiveSVM', color=tab10_colors[0], alpha=0.8)
        bars2 = axes[idx].bar(x - 0.5*width, sv_prob, width, label='ProbSVM', color=tab10_colors[1], alpha=0.8)
        bars3 = axes[idx].bar(x + 0.5*width, sv_knn, width, label='KNNSVM', color=tab10_colors[2], alpha=0.8)
        bars4 = axes[idx].bar(x + 1.5*width, sv_skip, width, label='SKiP', color=tab10_colors[3], alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                              f'{int(height)}',
                              ha='center', va='bottom', fontsize=8)
        
        axes[idx].set_xlabel('Outlier Level (Feature/Label)', fontsize=11)
        axes[idx].set_ylabel('Number of Support Vectors', fontsize=11)
        axes[idx].set_title(f'{dataset}', fontsize=12, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(noise_levels, rotation=45, ha='right')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Support Vector Count Comparison', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save PNG
    plt.savefig('visualizations/support_vector/support_vector_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: visualizations/support_vector/support_vector_comparison.png")
    
    # Save PDF
    plt.savefig('visualizations/support_vector/pdf/support_vector_comparison.pdf', bbox_inches='tight')
    print("✓ Saved: visualizations/support_vector/pdf/support_vector_comparison.pdf")
    
    plt.close('all')
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - support_vector_comparison.csv")
    print("  - visualizations/support_vector_comparison.png")
    print("  - visualizations/pdf/support_vector_comparison.pdf")


def print_summary_statistics(df):
    """
    Print summary statistics of the experiment
    
    Parameters:
    - df: DataFrame with experiment results
    """
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    # Overall statistics by model
    print("\nOverall Average Support Vectors by Model:")
    print(df.groupby('Model')['Support_Vectors'].agg(['mean', 'std', 'min', 'max']).round(2))
    
    print("\nOverall Average SV Ratio by Model:")
    print(df.groupby('Model')['SV_Ratio'].agg(['mean', 'std', 'min', 'max']).round(4))
    
    # Statistics by noise level
    df['Noise_Level'] = df['Feature_Noise'] + '/' + df['Label_Noise']
    
    print("\nAverage Support Vectors by Noise Level:")
    pivot_sv = df.pivot_table(values='Support_Vectors', index='Noise_Level', 
                               columns='Model', aggfunc='mean')
    print(pivot_sv.round(2))
    
    print("\nAverage SV Ratio by Noise Level:")
    pivot_ratio = df.pivot_table(values='SV_Ratio', index='Noise_Level', 
                                  columns='Model', aggfunc='mean')
    print(pivot_ratio.round(4))
    
    # Reduction percentage
    print("\nSV Reduction by SKiP-average compared to NaiveSVM (%):")
    for noise in ['0%/0%', '5%/5%', '10%/10%', '15%/15%', '20%/20%']:
        naive_avg = df[(df['Model'] == 'NaiveSVM') & (df['Noise_Level'] == noise)]['Support_Vectors'].mean()
        skip_avg = df[(df['Model'] == 'SKiP-average') & (df['Noise_Level'] == noise)]['Support_Vectors'].mean()
        reduction = ((naive_avg - skip_avg) / naive_avg) * 100
        print(f"  {noise}: {reduction:.2f}%")


def main():
    """Main function"""
    # Check if results file already exists
    if os.path.exists('support_vector_comparison.csv'):
        print("="*70)
        print("EXISTING RESULTS FOUND")
        print("="*70)
        print("\nLoading results from support_vector_comparison.csv...")
        df = pd.read_csv('support_vector_comparison.csv')
        print(f"Loaded {len(df)} experiment results.")
    else:
        # Run experiment
        df = run_experiment()
    
    # Print summary statistics
    print_summary_statistics(df)
    
    # Create visualizations
    visualize_results(df)
    
    print("\n" + "="*70)
    print("ALL TASKS COMPLETED SUCCESSFULLY")
    print("="*70)


if __name__ == '__main__':
    main()
