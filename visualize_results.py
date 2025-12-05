import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory for visualizations
output_dir = 'visualizations'
os.makedirs(output_dir, exist_ok=True)

# Load the CSV file
df_all_results = pd.read_csv('model_comparison_results.csv')

# Create heatmaps for each dataset showing noise impact on best model performance
for dataset_name in ['breast_cancer', 'breast_cancer_pca', 'iris', 'iris_pca', 'titanic', 'titanic_pca', 'wine', 'wine_pca']:
    dataset_results = df_all_results[df_all_results['Dataset'] == dataset_name]

    # Skip if dataset not found
    if dataset_results.empty:
        print(f"No data found for {dataset_name}, skipping...")
        continue

    # Get best performance for each noise combination and model
    best_per_combo = dataset_results.loc[
        dataset_results.groupby(['Feature_Noise', 'Label_Noise', 'Model'])['Test Acc'].idxmax()
    ]

    # Focus on main models - include all SKiP variants
    main_models = ['NaiveSVM', 'ProbSVM', 'KNNSVM', None,
                   'SKiP-multiply', 'SKiP-multiply-minmax',
                   'SKiP-average', 'SKiP-average-minmax']
    
    # Create separate figures for each kernel
    for kernel in ['linear', 'rbf']:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()

        for idx, model in enumerate(main_models):
            if model is None:
                axes[idx].set_visible(False)
                continue

            model_data = best_per_combo[(best_per_combo['Model'] == model) & 
                                       (best_per_combo['Kernel'] == kernel)]        # Create pivot table
        pivot = model_data.pivot_table(
            values='Test Acc',
            index='Label_Noise',
            columns='Feature_Noise',
            aggfunc='mean'
        )

        # Reorder columns and index
        feature_order = ['Clean', '5%', '10%', '15%', '20%']
        label_order = ['0%', '5%', '10%', '15%', '20%']
        pivot = pivot.reindex(index=label_order, columns=feature_order)

        # Plot heatmap
        ax = axes[idx]
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)

        # Set ticks
        ax.set_xticks(range(len(feature_order)))
        ax.set_yticks(range(len(label_order)))
        ax.set_xticklabels(feature_order)
        ax.set_yticklabels(['Clean' if label == '0%' else label for label in label_order])

        # Add text annotations
        for i in range(len(label_order)):
            for j in range(len(feature_order)):
                if not np.isnan(pivot.values[i, j]):
                    text = ax.text(j, i, f'{pivot.values[i, j]:.3f}',
                                 ha='center', va='center', color='black', fontsize=7)

            ax.set_title(f'{model}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Feature Noise', fontsize=9)
            ax.set_ylabel('Label Noise', fontsize=9)


        # Add colorbar
        cbar_ax = fig.add_axes([0.785, 0.53, 0.015, 0.4])  
        fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='Test Accuracy')

        plt.suptitle(f'Noise Impact on Test Accuracy - {dataset_name.upper()} ({kernel.upper()} Kernel)', 
                     fontsize=15, fontweight='bold')
        plt.tight_layout()
        output_path_png = os.path.join(output_dir, f'noise_heatmap_{dataset_name}_{kernel}.png')
        output_path_pdf = os.path.join(output_dir, f'noise_heatmap_{dataset_name}_{kernel}.pdf')
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf, bbox_inches='tight')
        plt.show()
        print(f"Saved heatmap for {dataset_name} ({kernel}) to {output_path_png} and {output_path_pdf}")