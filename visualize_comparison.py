import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directories for visualizations
output_dir = 'visualizations'
comparison_dir = os.path.join(output_dir, 'comparison')
comparison_pdf_dir = os.path.join(comparison_dir, 'pdf')

os.makedirs(comparison_dir, exist_ok=True)
os.makedirs(comparison_pdf_dir, exist_ok=True)

# Load the CSV file
df_all_results = pd.read_csv('model_comparison_results.csv')

target_models = ['NaiveSVM', 'SKiP-average']
df_skip = df_all_results[df_all_results['Model'].isin(target_models)]

# Get all unique datasets
datasets = df_skip['Dataset'].unique()

# Create comparison visualizations for each dataset
for dataset_name in datasets:
    dataset_results = df_skip[df_skip['Dataset'] == dataset_name]
    
    if dataset_results.empty:
        print(f"No data found for {dataset_name}, skipping...")
        continue
    
    # Get best performance for each noise combination and model
    best_per_combo = dataset_results.loc[
        dataset_results.groupby(['Feature_Noise', 'Label_Noise', 'Model', 'Kernel'])['Test Acc'].idxmax()
    ]
    
    # Create separate figures for each kernel
    for kernel in ['linear', 'rbf']:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Filter data for this kernel
        kernel_data = best_per_combo[best_per_combo['Kernel'] == kernel]
        
        if kernel_data.empty:
            print(f"No data found for {dataset_name} with {kernel} kernel, skipping...")
            plt.close()
            continue
        
        # 1. Side-by-side heatmaps
        for idx, model in enumerate(target_models):
            model_data = kernel_data[kernel_data['Model'] == model]
            
            # Create pivot table
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
                        text = ax.text(j, i, f'{pivot.values[i, j]*100:.1f}%',
                                     ha='center', va='center', color='black', fontsize=15)
            
            if model == "SKiP-average":
                display_model = "SKiP"
            else:
                display_model = model
            
            ax.set_title(f'{display_model}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Feature Noise', fontsize=10)
            ax.set_ylabel('Label Noise', fontsize=10)
        
        # 3. Difference heatmap (minmax - average)
        ax = axes[2]
        
        avg_data = kernel_data[kernel_data['Model'] == target_models[0]]
        minmax_data = kernel_data[kernel_data['Model'] == target_models[1]]
        
        avg_pivot = avg_data.pivot_table(
            values='Test Acc',
            index='Label_Noise',
            columns='Feature_Noise',
            aggfunc='mean'
        )
        
        minmax_pivot = minmax_data.pivot_table(
            values='Test Acc',
            index='Label_Noise',
            columns='Feature_Noise',
            aggfunc='mean'
        )
        
        # Reorder
        avg_pivot = avg_pivot.reindex(index=label_order, columns=feature_order)
        minmax_pivot = minmax_pivot.reindex(index=label_order, columns=feature_order)
        
        # Calculate difference
        diff_pivot = minmax_pivot - avg_pivot
        
        # Plot difference heatmap
        im_diff = ax.imshow(diff_pivot.values, cmap='RdBu_r', aspect='auto', 
                           vmin=-0.1, vmax=0.1)
        
        # Set ticks
        ax.set_xticks(range(len(feature_order)))
        ax.set_yticks(range(len(label_order)))
        ax.set_xticklabels(feature_order)
        ax.set_yticklabels(['Clean' if label == '0%' else label for label in label_order])
        
        # Add text annotations
        for i in range(len(label_order)):
            for j in range(len(feature_order)):
                if not np.isnan(diff_pivot.values[i, j]):
                    value = diff_pivot.values[i, j]
                    color = 'white' if abs(value) > 0.05 else 'black'
                    sign = '+' if value > 0 else ''
                    text = ax.text(j, i, f'{sign}{value*100:.1f}%',
                                 ha='center', va='center', color=color, fontsize=15)
        
        ax.set_title(f'Difference ({target_models[1].replace("SKiP-average", "SKiP")} - {target_models[0].replace("SKiP-average", "SKiP")})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Noise', fontsize=10)
        ax.set_ylabel('Label Noise', fontsize=10)
        
        # # Add colorbars
        # cbar1 = plt.colorbar(im, ax=axes[:2], orientation='vertical', 
        #                     label='Test Accuracy', pad=0.02, fraction=0.046)
        cbar2 = plt.colorbar(im_diff, ax=axes[2], orientation='vertical', 
                            label='Accuracy Difference', pad=0.02, fraction=0.046)
        
        plt.suptitle(f'{target_models[0].replace("SKiP-average", "SKiP")} vs {target_models[1].replace("SKiP-average", "SKiP")} Comparison\n{dataset_name.upper()} ({kernel.upper()} Kernel)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        output_path_png = os.path.join(comparison_dir, f'comparison_{dataset_name}_{kernel}.png')
        output_path_pdf = os.path.join(comparison_pdf_dir, f'comparison_{dataset_name}_{kernel}.pdf')
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison for {dataset_name} ({kernel}) to {output_path_png} and {output_path_pdf}")

# Create overall performance comparison across all datasets
print("\nCreating overall performance comparison...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

kernels = ['linear', 'rbf']
noise_levels = ['Clean', 'Low Noise', 'Medium Noise', 'High Noise']

# Define noise categories
def categorize_noise(row):
    feature = row['Feature_Noise']
    label = row['Label_Noise']
    
    if feature == 'Clean' and label == '0%':
        return 'Clean'
    elif feature in ['Clean', '5%'] and label in ['0%', '5%']:
        return 'Low Noise'
    elif feature in ['Clean', '5%', '10%'] and label in ['0%', '5%', '10%']:
        return 'Medium Noise'
    else:
        return 'High Noise'

df_skip['Noise_Category'] = df_skip.apply(categorize_noise, axis=1)

# Get best results per configuration
best_results = df_skip.loc[
    df_skip.groupby(['Dataset', 'Noise_Category', 'Model', 'Kernel'])['Test Acc'].idxmax()
]

for idx, kernel in enumerate(kernels):
    kernel_results = best_results[best_results['Kernel'] == kernel]
    
    # Average across datasets
    avg_by_noise = kernel_results.groupby(['Noise_Category', 'Model'])['Test Acc'].mean().reset_index()
    
    # Create bar chart
    ax = axes[idx]
    x = np.arange(len(noise_levels))
    width = 0.35
    
    avg_values = []
    minmax_values = []
    
    for noise in noise_levels:
        avg_val = avg_by_noise[(avg_by_noise['Noise_Category'] == noise) & 
                               (avg_by_noise['Model'] == target_models[0])]['Test Acc'].values
        minmax_val = avg_by_noise[(avg_by_noise['Noise_Category'] == noise) & 
                                  (avg_by_noise['Model'] == target_models[1])]['Test Acc'].values
        
        avg_values.append(avg_val[0] if len(avg_val) > 0 else 0)
        minmax_values.append(minmax_val[0] if len(minmax_val) > 0 else 0)
    
    bars1 = ax.bar(x - width/2, avg_values, width, label=target_models[0], alpha=0.8)
    bars2 = ax.bar(x + width/2, minmax_values, width, label=target_models[1], alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height*100:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Noise Level', fontsize=11)
    ax.set_ylabel('Average Test Accuracy', fontsize=11)
    ax.set_title(f'{kernel.upper()} Kernel - Average Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(noise_levels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.7, 1.0])
    
    # Difference plot
    ax = axes[idx + 2]
    differences = [minmax_values[i] - avg_values[i] for i in range(len(noise_levels))]
    colors = ['green' if d > 0 else 'red' for d in differences]
    
    bars = ax.bar(x, differences, color=colors, alpha=0.7)
    
    # Add value labels
    for i, (bar, diff) in enumerate(zip(bars, differences)):
        ax.text(bar.get_x() + bar.get_width()/2., diff,
               f'{diff*100:+.1f}%',
               ha='center', va='bottom' if diff > 0 else 'top', fontsize=15)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Noise Level', fontsize=11)
    ax.set_ylabel('Accuracy Difference (minmax - average)', fontsize=11)
    ax.set_title(f'{kernel.upper()} Kernel - Performance Difference', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(noise_levels)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Overall Performance Comparison', 
             fontsize=15, fontweight='bold')
plt.tight_layout()

output_path_png = os.path.join(comparison_dir, 'overall_comparison.png')
output_path_pdf = os.path.join(comparison_pdf_dir, 'overall_comparison.pdf')
plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
plt.savefig(output_path_pdf, bbox_inches='tight')
plt.close()

print(f"Saved overall comparison to {output_path_png} and {output_path_pdf}")



# === END ===


# Create separate visualization for iris_pca and wine_pca difference heatmaps
print("\nCreating iris_pca and wine_pca difference comparison...")

target_datasets = ['iris_pca', 'wine_pca']
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for dataset_idx, dataset_name in enumerate(target_datasets):
    dataset_results = df_skip[df_skip['Dataset'] == dataset_name]
    
    if dataset_results.empty:
        print(f"No data found for {dataset_name}, skipping...")
        continue
    
    # Get best performance for each noise combination and model
    best_per_combo = dataset_results.loc[
        dataset_results.groupby(['Feature_Noise', 'Label_Noise', 'Model', 'Kernel'])['Test Acc'].idxmax()
    ]
    
    kernel = 'linear'
    kernel_data = best_per_combo[best_per_combo['Kernel'] == kernel]
    
    if kernel_data.empty:
        print(f"No data found for {dataset_name} with {kernel} kernel, skipping...")
        continue
    
    ax = axes[dataset_idx]
        
    avg_data = kernel_data[kernel_data['Model'] == target_models[0]]
    minmax_data = kernel_data[kernel_data['Model'] == target_models[1]]
    
    avg_pivot = avg_data.pivot_table(
        values='Test Acc',
        index='Label_Noise',
        columns='Feature_Noise',
        aggfunc='mean'
    )
    
    minmax_pivot = minmax_data.pivot_table(
        values='Test Acc',
        index='Label_Noise',
        columns='Feature_Noise',
        aggfunc='mean'
    )
    
    # Reorder
    feature_order = ['Clean', '5%', '10%', '15%', '20%']
    label_order = ['0%', '5%', '10%', '15%', '20%']
    avg_pivot = avg_pivot.reindex(index=label_order, columns=feature_order)
    minmax_pivot = minmax_pivot.reindex(index=label_order, columns=feature_order)
    
    # Calculate difference
    diff_pivot = minmax_pivot - avg_pivot
    
    # Plot difference heatmap
    im_diff = ax.imshow(diff_pivot.values, cmap='RdBu_r', aspect='auto', 
                        vmin=-0.1, vmax=0.1)
    
    # Set ticks
    ax.set_xticks(range(len(feature_order)))
    ax.set_yticks(range(len(label_order)))
    ax.set_xticklabels(feature_order)
    ax.set_yticklabels(['Clean' if label == '0%' else label for label in label_order])
    
    # Add text annotations
    for i in range(len(label_order)):
        for j in range(len(feature_order)):
            if not np.isnan(diff_pivot.values[i, j]):
                value = diff_pivot.values[i, j]
                color = 'white' if abs(value) > 0.05 else 'black'
                sign = '+' if value > 0 else ''
                text = ax.text(j, i, f'{sign}{value*100:.1f}%',
                                ha='center', va='center', color=color, fontsize=15)
    
    ax.set_title(f'{dataset_name.upper()}\nDifference (SKiP - NaiveSVM)', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature Noise', fontsize=10)
    ax.set_ylabel('Label Noise', fontsize=10)
    
# Add colorbar
plt.colorbar(im_diff, ax=ax, orientation='vertical', 
            label='Accuracy Difference', pad=0.02, fraction=0.046)

plt.suptitle('Difference Heatmaps: SKiP vs NaiveSVM (Linear Kernel)', 
             fontsize=15, fontweight='bold')
plt.tight_layout()

output_path_png = os.path.join(comparison_dir, 'iris_wine_pca_difference.png')
output_path_pdf = os.path.join(comparison_pdf_dir, 'iris_wine_pca_difference.pdf')
plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
plt.savefig(output_path_pdf, bbox_inches='tight')
plt.close()

print(f"Saved iris_pca and wine_pca difference comparison to {output_path_png} and {output_path_pdf}")

# Create separate visualization for titanic_pca difference heatmap
print("\nCreating titanic_pca difference comparison...")

dataset_name = 'titanic_pca'
dataset_results = df_skip[df_skip['Dataset'] == dataset_name]

if not dataset_results.empty:
    # Get best performance for each noise combination and model
    best_per_combo = dataset_results.loc[
        dataset_results.groupby(['Feature_Noise', 'Label_Noise', 'Model', 'Kernel'])['Test Acc'].idxmax()
    ]
    
    kernel = 'linear'
    kernel_data = best_per_combo[best_per_combo['Kernel'] == kernel]
    
    if not kernel_data.empty:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        avg_data = kernel_data[kernel_data['Model'] == target_models[0]]
        minmax_data = kernel_data[kernel_data['Model'] == target_models[1]]
        
        avg_pivot = avg_data.pivot_table(
            values='Test Acc',
            index='Label_Noise',
            columns='Feature_Noise',
            aggfunc='mean'
        )
        
        minmax_pivot = minmax_data.pivot_table(
            values='Test Acc',
            index='Label_Noise',
            columns='Feature_Noise',
            aggfunc='mean'
        )
        
        # Reorder
        feature_order = ['Clean', '5%', '10%', '15%', '20%']
        label_order = ['0%', '5%', '10%', '15%', '20%']
        avg_pivot = avg_pivot.reindex(index=label_order, columns=feature_order)
        minmax_pivot = minmax_pivot.reindex(index=label_order, columns=feature_order)
        
        # Calculate difference
        diff_pivot = minmax_pivot - avg_pivot
        
        # Plot difference heatmap
        im_diff = ax.imshow(diff_pivot.values, cmap='RdBu_r', aspect='auto', 
                           vmin=-0.1, vmax=0.1)
        
        # Set ticks
        ax.set_xticks(range(len(feature_order)))
        ax.set_yticks(range(len(label_order)))
        ax.set_xticklabels(feature_order)
        ax.set_yticklabels(['Clean' if label == '0%' else label for label in label_order])
        
        # Add text annotations
        for i in range(len(label_order)):
            for j in range(len(feature_order)):
                if not np.isnan(diff_pivot.values[i, j]):
                    value = diff_pivot.values[i, j]
                    color = 'white' if abs(value) > 0.05 else 'black'
                    sign = '+' if value > 0 else ''
                    text = ax.text(j, i, f'{sign}{value*100:.1f}%',
                                 ha='center', va='center', color=color, fontsize=15)
        
        ax.set_title(f'{dataset_name.upper()}\nDifference (SKiP - NaiveSVM)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Noise', fontsize=10)
        ax.set_ylabel('Label Noise', fontsize=10)
        
        # Add colorbar
        plt.colorbar(im_diff, ax=ax, orientation='vertical', 
                    label='Accuracy Difference', pad=0.02, fraction=0.046)
        
        plt.suptitle('Difference Heatmap: SKiP vs NaiveSVM (Linear Kernel)', 
                     fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        output_path_png = os.path.join(comparison_dir, 'titanic_pca_difference.png')
        output_path_pdf = os.path.join(comparison_pdf_dir, 'titanic_pca_difference.pdf')
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf, bbox_inches='tight')
        plt.close()
        
        print(f"Saved titanic_pca difference comparison to {output_path_png} and {output_path_pdf}")
    else:
        print(f"No data found for {dataset_name} with linear kernel, skipping...")
else:
    print(f"No data found for {dataset_name}, skipping...")

print("\n" + "="*70)
print("COMPARISON VISUALIZATIONS COMPLETED")
print("="*70)
