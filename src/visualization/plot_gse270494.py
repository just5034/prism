"""
Visualization script for GSE270494 hematological malignancy methylation data.

This module creates publication-quality visualizations including:
- PCA scatter plots colored by disease type
- Methylation distribution histograms
- Hierarchical clustering heatmaps

Author: Generated for bioml project
Date: October 20, 2025
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loading import extract_methylation_matrix

# Set random seed for reproducibility
np.random.seed(42)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_processed_data():
    """
    Load processed PCA results and sample metadata.

    Returns
    -------
    df_pca : pd.DataFrame
        PCA results with PC1-PC5 and sample metadata.
    df_meta : pd.DataFrame
        Sample metadata with disease type annotations.
    """
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'

    df_pca = pd.read_csv(data_dir / 'GSE270494_PCA_results.csv')
    df_meta = pd.read_csv(data_dir / 'GSE270494_sample_metadata.csv')

    print(f"Loaded PCA results: {df_pca.shape}")
    print(f"Loaded metadata: {df_meta.shape}")
    print(f"\nDisease type distribution:")
    print(df_pca['disease_type'].value_counts())

    return df_pca, df_meta


def plot_pca_scatter(df_pca: pd.DataFrame, output_dir: Path):
    """
    Create PCA scatter plot (PC1 vs PC2) colored by disease type.

    Parameters
    ----------
    df_pca : pd.DataFrame
        PCA results with PC1, PC2, and disease_type columns.
    output_dir : Path
        Directory to save the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define color palette for disease types
    disease_types = df_pca['disease_type'].unique()
    disease_types = sorted([dt for dt in disease_types if dt != 'Unknown']) + ['Unknown']

    # Use a colorblind-friendly palette
    colors = sns.color_palette("husl", n_colors=len(disease_types) - 1)
    colors.append((0.7, 0.7, 0.7))  # Gray for Unknown
    color_map = dict(zip(disease_types, colors))

    # Plot each disease type separately for better legend control
    for disease in disease_types:
        subset = df_pca[df_pca['disease_type'] == disease]

        if disease == 'Unknown':
            ax.scatter(subset['PC1'], subset['PC2'],
                      c=[color_map[disease]], label=disease,
                      alpha=0.3, s=50, edgecolors='none')
        else:
            ax.scatter(subset['PC1'], subset['PC2'],
                      c=[color_map[disease]], label=disease,
                      alpha=0.7, s=100, edgecolors='black', linewidths=0.5)

    ax.set_xlabel('PC1 (29.23% variance explained)', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC2 (15.89% variance explained)', fontsize=12, fontweight='bold')
    ax.set_title('PCA of GSE270494 Hematological Malignancy Cell Lines\n(Top 10,000 Most Variable CpG Sites)',
                 fontsize=14, fontweight='bold', pad=20)

    # Place legend outside plot area
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
             frameon=True, fontsize=9, title='Disease Type', title_fontsize=10)

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = output_dir / 'GSE270494_PCA_scatter.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"\n[OK] Saved PCA scatter plot: {output_path}")

    plt.close()


def plot_pca_scree(df_pca: pd.DataFrame, output_dir: Path):
    """
    Create scree plot showing variance explained by each PC.

    Parameters
    ----------
    df_pca : pd.DataFrame
        PCA results.
    output_dir : Path
        Directory to save the plot.
    """
    # Variance explained (from Session 2 summary)
    variance_explained = [29.23, 15.89, 7.37, 3.12, 2.24]
    cumulative_variance = np.cumsum(variance_explained)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Individual variance
    ax1.bar(range(1, 6), variance_explained, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Principal Component', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Variance Explained (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Variance Explained by Each PC', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(1, 6))
    ax1.set_xticklabels([f'PC{i}' for i in range(1, 6)])
    ax1.grid(axis='y', alpha=0.3)

    # Add values on bars
    for i, v in enumerate(variance_explained):
        ax1.text(i+1, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Cumulative variance
    ax2.plot(range(1, 6), cumulative_variance, marker='o', linewidth=2,
            markersize=8, color='darkorange')
    ax2.fill_between(range(1, 6), cumulative_variance, alpha=0.3, color='orange')
    ax2.set_xlabel('Principal Component', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(1, 6))
    ax2.set_xticklabels([f'PC{i}' for i in range(1, 6)])
    ax2.grid(alpha=0.3)
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax2.legend()

    # Add values on points
    for i, v in enumerate(cumulative_variance):
        ax2.text(i+1, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    output_path = output_dir / 'GSE270494_PCA_scree.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"[OK] Saved scree plot: {output_path}")

    plt.close()


def plot_methylation_distributions(output_dir: Path):
    """
    Create methylation distribution histograms.

    Parameters
    ----------
    output_dir : Path
        Directory to save the plot.
    """
    print("\n--- Loading full methylation matrix for distribution analysis ---")

    # Load methylation data
    df_meth = extract_methylation_matrix('GSE270494', species='human',
                                         filter_detection_pvals=True)

    print(f"Methylation matrix shape: {df_meth.shape}")

    # Calculate statistics
    mean_meth = df_meth.values.flatten().mean()
    median_meth = np.median(df_meth.values.flatten())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Overall methylation distribution
    ax = axes[0, 0]
    ax.hist(df_meth.values.flatten(), bins=100, color='steelblue',
           alpha=0.7, edgecolor='black')
    ax.axvline(mean_meth, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_meth:.3f}')
    ax.axvline(median_meth, color='orange', linestyle='--', linewidth=2, label=f'Median = {median_meth:.3f}')
    ax.set_xlabel('Beta Value (Methylation Level)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency (log scale)', fontsize=11, fontweight='bold')
    ax.set_title('Global Methylation Distribution\n(All 760,090 CpG Sites × 180 Samples)',
                fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Per-sample mean methylation
    ax = axes[0, 1]
    sample_means = df_meth.mean(axis=0)
    ax.hist(sample_means, bins=30, color='darkgreen', alpha=0.7, edgecolor='black')
    ax.axvline(sample_means.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean = {sample_means.mean():.3f}')
    ax.set_xlabel('Mean Methylation per Sample', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Sample-Level Mean Methylation', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Per-CpG site mean methylation
    ax = axes[1, 0]
    cpg_means = df_meth.mean(axis=1)
    ax.hist(cpg_means, bins=100, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(cpg_means.mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean = {cpg_means.mean():.3f}')
    ax.set_xlabel('Mean Methylation per CpG Site', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of CpG Sites', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of CpG-Level Mean Methylation', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Methylation categories (hypo/intermediate/hyper)
    ax = axes[1, 1]
    flattened = df_meth.values.flatten()
    hypo = (flattened < 0.3).sum()
    inter = ((flattened >= 0.3) & (flattened <= 0.7)).sum()
    hyper = (flattened > 0.7).sum()

    categories = ['Hypomethylated\n(<0.3)', 'Intermediate\n(0.3-0.7)', 'Hypermethylated\n(>0.7)']
    counts = [hypo, inter, hyper]
    percentages = [c / len(flattened) * 100 for c in counts]
    colors_cat = ['lightcoral', 'gold', 'lightblue']

    bars = ax.bar(categories, percentages, color=colors_cat, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Percentage of Measurements (%)', fontsize=11, fontweight='bold')
    ax.set_title('Methylation Categories Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{pct:.1f}%\n({counts[percentages.index(pct)]:,})',
               ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()

    output_path = output_dir / 'GSE270494_methylation_distributions.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"[OK] Saved methylation distributions: {output_path}")

    plt.close()

    # Clean up large dataframe
    del df_meth


def plot_hierarchical_clustering(output_dir: Path):
    """
    Create hierarchical clustering heatmap for annotated samples.

    Parameters
    ----------
    output_dir : Path
        Directory to save the plot.
    """
    print("\n--- Creating hierarchical clustering heatmap ---")

    # Load methylation data
    df_meth = extract_methylation_matrix('GSE270494', species='human',
                                         filter_detection_pvals=True)

    # Load metadata to filter annotated samples only
    df_meta = pd.read_csv(Path(__file__).parent.parent.parent / 'data' / 'processed' /
                         'GSE270494_sample_metadata.csv')

    # Filter to annotated samples only (exclude Unknown)
    annotated_samples = df_meta[df_meta['disease_type'] != 'Unknown']['sample_name'].values
    print(f"Using {len(annotated_samples)} annotated samples for clustering")

    # Get sample columns that match annotated samples
    sample_cols = [col for col in df_meth.columns if col in annotated_samples]
    df_meth_annotated = df_meth[sample_cols]

    print(f"Annotated methylation matrix: {df_meth_annotated.shape}")

    # Select top 1000 most variable CpG sites for visualization
    variances = df_meth_annotated.var(axis=1)
    top_variable_cpgs = variances.nlargest(1000).index
    df_meth_subset = df_meth_annotated.loc[top_variable_cpgs]

    print(f"Using top 1000 variable CpG sites for heatmap")

    # Standardize features (Z-score normalization)
    scaler = StandardScaler()
    arr_scaled = scaler.fit_transform(df_meth_subset.T).T
    df_scaled = pd.DataFrame(arr_scaled, index=df_meth_subset.index,
                            columns=df_meth_subset.columns)

    # Create disease type color map for columns
    disease_map = df_meta.set_index('sample_name')['disease_type'].to_dict()
    col_colors = pd.Series([disease_map.get(col, 'Unknown') for col in df_scaled.columns],
                          index=df_scaled.columns)

    # Define color palette
    disease_types = sorted(df_meta[df_meta['disease_type'] != 'Unknown']['disease_type'].unique())
    colors = sns.color_palette("husl", n_colors=len(disease_types))
    disease_color_map = dict(zip(disease_types, colors))

    col_colors_mapped = col_colors.map(disease_color_map)

    # Create clustermap
    print("Generating clustermap (this may take a minute)...")

    g = sns.clustermap(df_scaled,
                      col_colors=col_colors_mapped,
                      cmap='RdBu_r',
                      center=0,
                      vmin=-3, vmax=3,
                      figsize=(16, 12),
                      cbar_kws={'label': 'Z-score (Methylation)'},
                      xticklabels=True,
                      yticklabels=False,
                      method='ward',
                      metric='euclidean',
                      dendrogram_ratio=0.15,
                      cbar_pos=(0.02, 0.8, 0.03, 0.15))

    # Adjust labels
    g.ax_heatmap.set_xlabel('Samples (n=71 annotated)', fontsize=12, fontweight='bold')
    g.ax_heatmap.set_ylabel('CpG Sites (n=1,000 most variable)', fontsize=12, fontweight='bold')

    # Rotate x-axis labels
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)

    # Add title
    g.fig.suptitle('Hierarchical Clustering of GSE270494 Annotated Samples\n' +
                   'Based on Top 1,000 Most Variable CpG Sites',
                   fontsize=14, fontweight='bold', y=0.98)

    # Create custom legend for disease types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=disease_color_map[dt], label=dt)
                      for dt in disease_types]
    g.ax_col_dendrogram.legend(handles=legend_elements,
                              title='Disease Type',
                              loc='upper left',
                              bbox_to_anchor=(0, 1),
                              frameon=True,
                              fontsize=8,
                              title_fontsize=9)

    output_path = output_dir / 'GSE270494_hierarchical_clustering.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"[OK] Saved hierarchical clustering heatmap: {output_path}")

    plt.close()

    # Clean up
    del df_meth, df_meth_annotated, df_meth_subset


def main():
    """
    Main function to generate all visualizations.
    """
    print("=" * 80)
    print("GSE270494 VISUALIZATION PIPELINE")
    print("=" * 80)

    # Setup output directory
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load processed data
    print("\n" + "=" * 80)
    print("LOADING PROCESSED DATA")
    print("=" * 80)
    df_pca, df_meta = load_processed_data()

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    print("\n[1/5] PCA scatter plot...")
    plot_pca_scatter(df_pca, output_dir)

    print("\n[2/5] PCA scree plot...")
    plot_pca_scree(df_pca, output_dir)

    print("\n[3/5] Methylation distributions...")
    plot_methylation_distributions(output_dir)

    print("\n[4/5] Hierarchical clustering heatmap...")
    plot_hierarchical_clustering(output_dir)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nAll plots saved to: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob('*.png')):
        print(f"  - {file.name}")


if __name__ == '__main__':
    main()
