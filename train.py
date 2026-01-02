#!/usr/bin/env python3
"""
train_rf.py - Random Forest Classifier for PE Features
-----------------------------------------------------
Reads a CSV of extracted PE features (with a 'label' column),
trains a Random Forest classifier using scikit-learn,
evaluates accuracy on a held-out test set,
and saves the trained model as a pickle file.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # Load dataset
    try:
        df = pd.read_csv("pe_features.csv")
    except FileNotFoundError:
        print("Error: 'pe_features.csv' not found.")
        return

    X = df.drop(columns=["label"])
    y = df["label"]

    # Convert to numpy arrays to avoid feature name warnings during classification
    X = X.values
    y = y.values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # Initialize and train Random Forest
    # n_estimators is the number of trees in the forest
    rf_model = RandomForestClassifier(n_estimators=3, random_state=42, min_impurity_decrease=0.001) # min_impurity_decrease was set so I could reuse injector_1.cxx
    rf_model.fit(X_train, y_train)

    # Evaluation
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model
    model_filename = "random_forest_model.pkl"
    joblib.dump(rf_model, model_filename)
    print(f"\nModel saved to {model_filename}")

if __name__ == "__main__":
    main()