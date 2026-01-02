import joblib
import numpy as np
import os


def analyze_evasion_paths(model_path):
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    # Load the trained Random Forest
    rf_model = joblib.load(model_path)
    feature_names = ['weighted_entropy', 'strings_density', 'log_size']

    print(f"--- Evasion Analysis for {model_path} ---")
    print(f"Forest contains {len(rf_model.estimators_)} trees.")
    print("Goal: Identify feature constraints for Benign (Class 0) classification.\n")

    for i, estimator in enumerate(rf_model.estimators_):
        print(f"Tree {i}:")
        tree = estimator.tree_

        # Identify all leaf nodes
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        value = tree.value

        # Find leaves where class 0 (Benign) is the majority
        # value[node_id][0] gives the distribution [count_class_0, count_class_1]
        benign_leaf_ids = [
            node_id for node_id in range(tree.node_count)
            if children_left[node_id] == -1 and np.argmax(value[node_id][0]) == 0
        ]

        for leaf_id in benign_leaf_ids:
            # Trace path from leaf back to root
            path_constraints = []
            curr = leaf_id

            while curr != 0:  # 0 is the root node
                # Find parent
                parent = -1
                direction = ""

                # Check if current node is a left or right child of some parent
                for node_id in range(tree.node_count):
                    if children_left[node_id] == curr:
                        parent = node_id
                        direction = "<="
                        break
                    if children_right[node_id] == curr:
                        parent = node_id
                        direction = ">"
                        break

                fname = feature_names[feature[parent]]
                thresh = threshold[parent]
                path_constraints.append(f"{fname} {direction} {thresh:.4f}")
                curr = parent

            # Reverse to show root-to-leaf order
            path_str = " AND ".join(reversed(path_constraints))
            print(f"  [Leaf {leaf_id}] Path to Benign: {path_str}")
        print("-" * 30)


if __name__ == "__main__":
    analyze_evasion_paths("random_forest_model.pkl")