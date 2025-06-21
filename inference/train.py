"""Train a quick classifier on inference data with hand-crafted features"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def main(opts):
    # Load data from disk in csv
    path = os.path.join(
        "inference", "scores", f"{opts.dataset}_{opts.ftdata}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)

    # Dataset inspection
    print("Data shape:", df.shape)
    print("Class distribution", df["label"].value_counts())
    print(df.sample(5))

    # Prepare features and labels
    features = df.drop(columns=["label"])
    labels = df["label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model training
    model = LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced', C=10.)
    model.fit(X_train, y_train)
    print("\nTraining report:")
    print(classification_report(y_train, model.predict(X_train)))

    # Evaluation
    y_pred = model.predict(X_test)
    print("Test report:")
    print(classification_report(y_test, y_pred))

    # ROC curve

    # Save the model
    import joblib
    output_dir = "inference/models"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{opts.dataset}_{opts.ftdata}_clf.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ftdata", choices=["mixed", "ibm"],
                        default="mixed",
                        help="Choose classifier")
    parser.add_argument("--dataset", choices=["molecular", "thoracic", "merge"],
                        help="Choose dataset to train on")
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        import ipdb
        import traceback
        import sys
        print("Exception:", e)
        traceback.print_exc()
        ipdb.post_mortem(sys.exc_info()[2])
