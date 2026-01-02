#!/usr/bin/env python3
"""
visualize_forest.py
-------------------
Loads the trained Random Forest model and exports all trees
into a single PNG image using a subplot grid.
"""

import joblib
import os
import sys

try:
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
except ImportError:
    print("Error: 'matplotlib' is required for visualization.")
    print("Please install it using: pip install matplotlib")
    sys.exit(1)

def main():
    model_path = "random_forest_model.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    # Load the model
    print(f"Loading model from {model_path}...")
    rf_model = joblib.load(model_path)

    n_trees = len(rf_model.estimators_)
    print(f"Found {n_trees} trees. Generating forest overview...")

    # Define feature and class names
    feature_names = ['weighted_entropy', 'strings_density', 'log_size']
    class_names = ['BENIGN', 'MALWARE']

    # Calculate grid dimensions (e.g., for 3 trees, 1x3 or 2x2)
    cols = 3
    rows = (n_trees + cols - 1) // cols

    # Create a large figure to hold all trees
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 30, rows * 16))

    # Flatten axes array if it's 2D, or wrap in list if 1D
    if n_trees == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(n_trees):
        tree = rf_model.estimators_[i]
        plot_tree(tree,
                  node_ids=True,
                  feature_names=feature_names,
                  class_names=class_names,
                  filled=True,
                  rounded=True,
                  fontsize=8,
                  ax=axes[i])
        axes[i].set_title(f"Tree {i}")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    output_combined = "entire_forest.png"
    print(f"Saving combined visualization to {output_combined}...")
    plt.tight_layout(pad=5.0)
    plt.savefig(output_combined, dpi=200)
    plt.close()

    # Save individual trees
    output_dir = "forest_trees"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving {n_trees} individual trees to '{output_dir}/'...")
    for i in range(n_trees):
        plt.figure(figsize=(50, 20))
        plot_tree(rf_model.estimators_[i],
                  node_ids=True,
                  feature_names=feature_names,
                  class_names=class_names,
                  filled=True,
                  rounded=True,
                  fontsize=10)
        plt.title(f"Random Forest - Tree {i}")

        individual_file = os.path.join(output_dir, f"tree_{i}.png")
        plt.savefig(individual_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" - Saved {individual_file}")

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()