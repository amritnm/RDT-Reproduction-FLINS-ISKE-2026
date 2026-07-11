import pandas as pd
import os
from pathlib import Path

# Base directory (relative to this script's location)
base_dir = Path(__file__).resolve().parent / "Final_Results" / "Tree_Results"

# Results storage
results = []

# Iterate through all dataset folders
for dataset_dir in sorted(base_dir.iterdir()):
    if dataset_dir.is_dir() and dataset_dir.name.startswith("results_"):
        dataset_name = dataset_dir.name.replace("results_", "").replace("_", " ")
        summary_file = dataset_dir / "results_summary.csv"
        
        if summary_file.exists():
            try:
                df = pd.read_csv(summary_file)
                
                # Filter for depth 5, target encoding (more realistic for production)
                depth9 = df[(df['depth'] == 5) & (df['encoding'] == 'target')]
                
                if len(depth9) > 0:
                    # Extract results for each algorithm
                    sklearn_row = depth9[depth9['algorithm'] == 'sklearn-DT']
                    rdt_row = depth9[depth9['algorithm'] == 'RDT']
                    odt_row = depth9[depth9['algorithm'] == 'ODT']
                    contree_row = depth9[depth9['algorithm'] == 'ConTree']
                    
                    result = {
                        'Dataset': dataset_name,
                        'sklearn_acc': sklearn_row['mean_accuracy'].values[0] if len(sklearn_row) > 0 else None,
                        'sklearn_feat': sklearn_row['features_used'].values[0] if len(sklearn_row) > 0 else None,
                        'RDT_acc': rdt_row['mean_accuracy'].values[0] if len(rdt_row) > 0 else None,
                        'RDT_feat': rdt_row['features_used'].values[0] if len(rdt_row) > 0 else None,
                        'ODT_acc': odt_row['mean_accuracy'].values[0] if len(odt_row) > 0 else None,
                        'ConTree_acc': contree_row['mean_accuracy'].values[0] if len(contree_row) > 0 else None,
                    }
                    results.append(result)
                    print(f"[OK] {dataset_name}: RDT={result['RDT_acc']:.3f}, sklearn={result['sklearn_acc']:.3f}")
                else:
                    print(f"[SKIP] {dataset_name}: No depth 9 results")
            except Exception as e:
                print(f"[ERROR] {dataset_name}: Error - {e}")

# Create DataFrame and save
results_df = pd.DataFrame(results)
print(f"\n{'='*60}")
print(f"Total datasets with results: {len(results_df)}")
print(f"{'='*60}\n")

# Print formatted table for LaTeX
print("LaTeX Table Format:")
print("-" * 60)
for _, row in results_df.iterrows():
    rdt_pct = (row['RDT_acc'] / row['sklearn_acc'] * 100) if row['sklearn_acc'] else 0
    feat_reduction = (1 - row['RDT_feat'] / row['sklearn_feat']) * 100 if row['sklearn_feat'] else 0
    print(f"{row['Dataset']} & {row['sklearn_acc']:.3f} & {row['ODT_acc']:.3f} & "
          f"{row['ConTree_acc']:.3f} & {row['RDT_acc']:.3f} & "
          f"{int(row['RDT_feat'])} vs {int(row['sklearn_feat'])} & {rdt_pct:.1f}\\% \\\\")

# Calculate average feature reduction
avg_rdt_feat = results_df['RDT_feat'].mean()
avg_sklearn_feat = results_df['sklearn_feat'].mean()
avg_reduction = (1 - avg_rdt_feat / avg_sklearn_feat) * 100
print(f"\n{'='*60}")
print(f"Average RDT features: {avg_rdt_feat:.1f}")
print(f"Average sklearn features: {avg_sklearn_feat:.1f}")
print(f"Average reduction: {avg_reduction:.1f}%")
print(f"{'='*60}")

# Save to CSV
output_file = base_dir.parent.parent / "all_tree_results_summary.csv"
results_df.to_csv(output_file, index=False)
print(f"\n✓ Results saved to: {output_file}")
