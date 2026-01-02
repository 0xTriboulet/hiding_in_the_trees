#!/usr/bin/env python3
"""
classify.py
-----------
Loads the trained Random Forest model,
extracts features from a target PE binary,
and performs malware classification.
"""

import joblib
import numpy as np
import sys
import os
import math
import pefile
import string

# --------------------------------------------------
# Feature Extraction Implementation
# --------------------------------------------------
def shannon_entropy(data: bytes) -> float:
    """Compute Shannon entropy of a byte sequence."""
    if len(data) == 0:
        return 0.0
    freq = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = freq / len(data)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def extract_features(binary_path):
    """
    Extract PE features:
      - Weighted section entropy
      - Strings density
      - Log10 file size
    Returns: np.array([entropy, strings_density, log_size])
    """
    if not os.path.exists(binary_path):
        raise FileNotFoundError(f"File not found: {binary_path}")

    # --------------------------------------------------
    # File size
    # --------------------------------------------------
    file_size = os.path.getsize(binary_path)
    file_size_kb = max(file_size / 1024.0, 1e-6)  # avoid div-by-zero
    log_size = math.log10(file_size + 1)

    # --------------------------------------------------
    # Section entropy (size-weighted)
    # --------------------------------------------------
    try:
        pe = pefile.PE(binary_path, fast_load=True)
        section_entropies = []
        section_sizes = []
        for section in pe.sections:
            data = section.get_data()
            entropy = shannon_entropy(data)
            section_entropies.append(entropy)
            section_sizes.append(len(data))
        if section_sizes:
            weighted_entropy = np.average(section_entropies, weights=section_sizes)
        else:
            weighted_entropy = 0.0
    except Exception:
        weighted_entropy = 0.0

    # --------------------------------------------------
    # Strings density
    # --------------------------------------------------
    min_len = 4
    with open(binary_path, "rb") as f:
        raw_bytes = f.read()

    printable = set(bytes(string.printable, "ascii"))
    count_strings = 0
    current = bytearray()
    for b in raw_bytes:
        if b in printable and b not in b"\r\n\t":
            current.append(b)
        else:
            if len(current) >= min_len:
                count_strings += 1
            current = bytearray()
    if len(current) >= min_len:
        count_strings += 1

    strings_density = count_strings / file_size_kb

    # --------------------------------------------------
    # Return feature vector
    # --------------------------------------------------
    features = np.array([weighted_entropy, strings_density, log_size], dtype=np.float32)
    return features

# --------------------------------------------------
# Classification Logic
# --------------------------------------------------
def classify(binary_path):
    # Load model
    model_path = "random_forest_model.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    rf_model = joblib.load(model_path)

    # Extract features
    raw_features = extract_features(binary_path).reshape(1, -1)

    # Feature names for printing
    feature_names = ['weighted_entropy', 'strings_density', 'log_size']

    # Run model
    # predict_proba returns [prob_class_0, prob_class_1]
    prob = rf_model.predict_proba(raw_features)[0][1]
    label = rf_model.predict(raw_features)[0]

    verdict = "MALWARE" if label == 1 else "BENIGN"

    print(f"File: {os.path.basename(binary_path)}")
    print("Extracted Features:")
    for name, value in zip(feature_names, raw_features[0]):
        print(f"  - {name}: {value:.4f}")
    print(f"Probability of malware: {prob:.4f}")
    print(f"Classification: {verdict}")

    # Decision Path Visualization (text)
    print("\nDecision Paths across Forest:")
    for i, tree in enumerate(rf_model.estimators_):
        print(f"\n[Tree {i}] decision path:")
        node_indicator = tree.decision_path(raw_features)
        node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

        feature = tree.tree_.feature
        threshold = tree.tree_.threshold

        for node_id in node_index:
            if feature[node_id] != -2:  # -2 means it's a leaf node
                f_idx = feature[node_id]
                val = raw_features[0][f_idx]
                name = feature_names[f_idx]
                direction = "<=" if val <= threshold[node_id] else ">"
                print(f"  Node {node_id}: {name} ({val:.4f}) {direction} {threshold[node_id]:.4f}")
            else:
                leaf_value = tree.tree_.value[node_id]
                # Random Forest leaves store class distributions
                print(f"  Leaf {node_id}: Distribution {leaf_value[0]}")


# --------------------------------------------------
# Entry Point
# --------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <target_binary>")
        sys.exit(1)

    target_binary = sys.argv[1]
    classify(target_binary)