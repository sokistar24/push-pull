# file: code/train_basic_accelerometer_dt.py

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

FULL_FEATURE_ACCURACY = 0.9642  # From previous full-feature results


def prepare_basic_features(df: pd.DataFrame):
    """
    Prepare only basic accelerometer features (accx, accy, accz).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with basic accelerometer readings.
    y : pd.Series
        Target activity labels.
    feature_names : list[str]
        List of basic feature names actually used.
    """
    print("=== Basic Feature Preparation ===")

    basic_features = ["accx", "accy", "accz"]

    # Verify features exist in dataset
    available_features = [col for col in basic_features if col in df.columns]
    missing_features = [col for col in basic_features if col not in df.columns]

    print(f"Using basic features: {available_features}")
    if missing_features:
        print(f"Missing features in dataset: {missing_features}")
        return None, None, None

    X = df[available_features]
    y = df["activity"]

    print(f"Basic feature matrix shape: {X.shape}")
    print(f"Target classes: {sorted(y.unique())}")
    print(f"Class distribution: {y.value_counts().sort_index().to_dict()}")

    return X, y, available_features


def train_basic_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
):
    """
    Train a decision tree using only basic features and simple hyperparameter search.

    Returns
    -------
    best_classifier : DecisionTreeClassifier
        Trained classifier with best validation accuracy.
    best_params : dict
        Hyperparameters of the best model.
    """
    print("\n=== Basic Feature Decision Tree Training ===")

    hyperparams = [
        {"max_depth": 8, "min_samples_split": 20, "min_samples_leaf": 10},
        {"max_depth": 12, "min_samples_split": 30, "min_samples_leaf": 15},
        {"max_depth": 15, "min_samples_split": 50, "min_samples_leaf": 20},
        {"max_depth": None, "min_samples_split": 40, "min_samples_leaf": 15},
    ]

    best_score = 0.0
    best_params = None
    best_classifier = None

    for params in hyperparams:
        clf = DecisionTreeClassifier(random_state=42, **params)
        clf.fit(X_train, y_train)
        val_score = clf.score(X_val, y_val)

        print(f"Params: {params}")
        print(f"Validation accuracy: {val_score:.4f}\n")

        if val_score > best_score:
            best_score = val_score
            best_params = params
            best_classifier = clf

    print(f"Best parameters: {best_params}")
    print(f"Best validation accuracy: {best_score:.4f}")

    return best_classifier, best_params


def compare_models(basic_accuracy: float, full_accuracy: float) -> str:
    """
    Compare basic vs full feature model performance.

    Returns
    -------
    comparison_text : str
        Summary of the comparison.
    """
    print("\n=== Model Comparison ===")
    print(f"Basic features (accx, accy, accz) accuracy: {basic_accuracy:.4f}")
    print(f"Full features (all) accuracy: {full_accuracy:.4f}")

    accuracy_diff = full_accuracy - basic_accuracy
    performance_loss = accuracy_diff / full_accuracy * 100

    print(f"Accuracy difference: {accuracy_diff:.4f}")
    print(f"Performance loss relative to full model: {performance_loss:.2f}%")

    if performance_loss < 3:
        message = (
            "Basic features provide comparable performance and are suitable "
            "for edge deployment."
        )
    elif performance_loss < 8:
        message = (
            "There is moderate performance loss. Consider the trade-off "
            "between speed/resource use and accuracy."
        )
    else:
        message = (
            "There is significant performance loss. Engineered features "
            "provide substantial benefit."
        )

    print(message)

    comparison_text = (
        f"Basic accuracy: {basic_accuracy:.4f}\n"
        f"Full accuracy:  {full_accuracy:.4f}\n"
        f"Accuracy difference: {accuracy_diff:.4f}\n"
        f"Performance loss (% of full): {performance_loss:.2f}%\n"
        f"Interpretation: {message}"
    )
    return comparison_text


def analyze_basic_model_edge_suitability(classifier: DecisionTreeClassifier) -> str:
    """
    Analyse basic model for edge device deployment.

    Returns
    -------
    summary : str
        Multi-line summary of suitability.
    """
    print("\n=== Edge Device Suitability (Basic Model) ===")
    lines = []

    depth = classifier.tree_.max_depth
    node_count = classifier.tree_.node_count
    n_leaves = classifier.tree_.n_leaves
    approx_memory_kb = node_count * 8 * 3 / 1024  # Rough estimate

    header = [
        "Input features: 3 (accx, accy, accz)",
        f"Tree depth: {depth}",
        f"Number of nodes: {node_count}",
        f"Number of leaves: {n_leaves}",
        "",
        "Computational requirements:",
        "- Feature computation: None (raw sensor values only)",
        f"- Approximate maximum number of comparisons: {node_count}",
        f"- Approximate memory footprint: {approx_memory_kb:.1f} KB",
    ]
    lines.extend(header)

    if node_count < 50:
        suitability = (
            "Suitability: Excellent for edge deployment; inference is very fast."
        )
    elif node_count < 150:
        suitability = (
            "Suitability: Good for edge deployment; inference is fast "
            "for most devices."
        )
    else:
        suitability = (
            "Suitability: May require optimisation for very resource-constrained "
            "devices."
        )

    print()
    print(suitability)
    lines.append("")
    lines.append(suitability)

    return "\n".join(lines)


def visualize_basic_model_results(
    classifier: DecisionTreeClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    feature_names,
    results_dir: Path,
):
    """
    Visualise basic model performance and save to Results directory.

    Saves
    -----
    - basic_model_confusion_matrix.png
    - basic_model_feature_importance.png
    - basic_model_tree.png
    - basic_model_3d_scatter.png
    - basic_model_all_plots.png
    """
    activity_names = ["Idle/Sitting", "Standing", "Walking"]

    # 1. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=activity_names,
        yticklabels=activity_names,
    )
    plt.title("Confusion Matrix (Basic Features)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = results_dir / "basic_model_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to: {cm_path}")

    # 2. Feature importance
    importance_df = (
        pd.DataFrame(
            {"feature": feature_names, "importance": classifier.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    plt.figure(figsize=(6, 4))
    plt.bar(importance_df["feature"], importance_df["importance"])
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Feature Importance (Basic Features)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fi_path = results_dir / "basic_model_feature_importance.png"
    plt.savefig(fi_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance plot to: {fi_path}")

    # 3. Decision tree visualisation (truncated depth)
    plt.figure(figsize=(12, 8))
    plot_tree(
        classifier,
        max_depth=4,
        feature_names=feature_names,
        class_names=activity_names,
        filled=True,
    )
    plt.title("Decision Tree Structure (Basic Features, depth<=4)")
    plt.tight_layout()
    tree_path = results_dir / "basic_model_tree.png"
    plt.savefig(tree_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved decision tree plot to: {tree_path}")

    # 4. 3D scatter of correct vs incorrect classifications
    fig = plt.figure(figsize=(8, 6))
    ax_3d = fig.add_subplot(111, projection="3d")

    correct_mask = y_test.values == y_pred
    incorrect_mask = ~correct_mask

    if np.sum(correct_mask) > 0:
        correct_samples = X_test[correct_mask]
        # Subsample to keep plot readable
        if len(correct_samples) > 200:
            correct_samples = correct_samples.sample(n=200, random_state=42)
        ax_3d.scatter(
            correct_samples["accx"],
            correct_samples["accy"],
            correct_samples["accz"],
            c="green",
            alpha=0.6,
            label="Correct",
            s=5,
        )

    if np.sum(incorrect_mask) > 0:
        incorrect_samples = X_test[incorrect_mask]
        ax_3d.scatter(
            incorrect_samples["accx"],
            incorrect_samples["accy"],
            incorrect_samples["accz"],
            c="red",
            alpha=0.8,
            label="Incorrect",
            s=15,
        )

    ax_3d.set_xlabel("AccX")
    ax_3d.set_ylabel("AccY")
    ax_3d.set_zlabel("AccZ")
    ax_3d.set_title("Classification Results in 3D Space")
    ax_3d.legend()
    plt.tight_layout()
    scatter_path = results_dir / "basic_model_3d_scatter.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved 3D scatter plot to: {scatter_path}")

    # Optional combined figure (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Confusion matrix (again, for combined figure)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=activity_names,
        yticklabels=activity_names,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Confusion Matrix (Basic Features)")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("Actual")

    # Feature importance
    axes[0, 1].bar(importance_df["feature"], importance_df["importance"])
    axes[0, 1].set_xlabel("Feature")
    axes[0, 1].set_ylabel("Importance")
    axes[0, 1].set_title("Feature Importance (Basic Features)")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Decision tree (truncated)
    plot_tree(
        classifier,
        max_depth=4,
        feature_names=feature_names,
        class_names=activity_names,
        filled=True,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Decision Tree Structure (Basic Features)")

    # 3D scatter reused
    ax_3d_combined = fig.add_subplot(2, 2, 4, projection="3d")

    if np.sum(correct_mask) > 0:
        correct_samples = X_test[correct_mask]
        if len(correct_samples) > 200:
            correct_samples = correct_samples.sample(n=200, random_state=42)
        ax_3d_combined.scatter(
            correct_samples["accx"],
            correct_samples["accy"],
            correct_samples["accz"],
            c="green",
            alpha=0.6,
            label="Correct",
            s=5,
        )

    if np.sum(incorrect_mask) > 0:
        incorrect_samples = X_test[incorrect_mask]
        ax_3d_combined.scatter(
            incorrect_samples["accx"],
            incorrect_samples["accy"],
            incorrect_samples["accz"],
            c="red",
            alpha=0.8,
            label="Incorrect",
            s=15,
        )

    ax_3d_combined.set_xlabel("AccX")
    ax_3d_combined.set_ylabel("AccY")
    ax_3d_combined.set_zlabel("AccZ")
    ax_3d_combined.set_title("Classification Results in 3D Space")
    ax_3d_combined.legend()

    plt.tight_layout()
    combined_path = results_dir / "basic_model_all_plots.png"
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined plots to: {combined_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for data path and results directory.
    """
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]  # assuming script is in code/

    default_data_path = project_root / "data" / "accelerometer_dataset.csv"
    default_results_dir = project_root / "Results"

    parser = argparse.ArgumentParser(
        description=(
            "Train a decision tree classifier using only basic accelerometer "
            "features (accx, accy, accz)."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=default_data_path,
        help=f"Path to accelerometer CSV file (default: {default_data_path})",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir,
        help=f"Directory to save results (default: {default_results_dir})",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    data_path: Path = args.data_path
    results_dir: Path = args.results_dir

    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading extended dataset for basic feature training...")
    print(f"Data path: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    extended_data = pd.read_csv(data_path)
    print(f"Dataset loaded: {extended_data.shape}")

    # Prepare basic features
    X_basic, y, basic_feature_names = prepare_basic_features(extended_data)

    if X_basic is None:
        print("Basic features could not be prepared. Exiting.")
        return

    # Train/validation/test split
    X_train_basic, X_temp, y_train, y_temp = train_test_split(
        X_basic,
        y,
        test_size=0.4,
        random_state=42,
        stratify=y,
    )
    X_val_basic, X_test_basic, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )

    print("\nBasic feature data splits:")
    print(f"Training:   {X_train_basic.shape[0]:,} samples")
    print(f"Validation: {X_val_basic.shape[0]:,} samples")
    print(f"Test:       {X_test_basic.shape[0]:,} samples")

    # Train basic classifier
    basic_classifier, basic_params = train_basic_classifier(
        X_train_basic,
        y_train,
        X_val_basic,
        y_val,
    )

    # Evaluate basic model
    y_pred_basic = basic_classifier.predict(X_test_basic)
    basic_accuracy = accuracy_score(y_test, y_pred_basic)

    print("\n=== Basic Model Evaluation ===")
    print(f"Basic model test accuracy: {basic_accuracy:.4f}")

    activity_names = ["Idle/Sitting", "Standing", "Walking"]
    classification_rep = classification_report(
        y_test, y_pred_basic, target_names=activity_names
    )
    print("\nClassification Report (Basic Features):")
    print(classification_rep)

    # Cross-validation on full dataset with best params
    print("\n=== Cross-Validation (Basic Features) ===")
    clf_cv = DecisionTreeClassifier(random_state=42, **basic_params)
    cv_scores = cross_val_score(clf_cv, X_basic, y, cv=5, scoring="accuracy")
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    print(f"5-fold CV mean accuracy: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")

    # Compare with full feature model
    comparison_text = compare_models(basic_accuracy, FULL_FEATURE_ACCURACY)

    # Analyse edge suitability
    edge_suitability_text = analyze_basic_model_edge_suitability(basic_classifier)

    # Visualise and save plots
    visualize_basic_model_results(
        basic_classifier,
        X_test_basic,
        y_test,
        y_pred_basic,
        basic_feature_names,
        results_dir,
    )

    # Save model
    model_path = results_dir / "accelerometer_decision_tree_basic_features.pkl"
    joblib.dump(basic_classifier, model_path)
    print(f"\nBasic model saved to: {model_path}")

    # Save textual report
    report_path = results_dir / "basic_model_report.txt"
    report_lines = [
        "=== Basic Accelerometer Decision Tree Model (Basic Features) ===",
        "",
        f"Data path: {data_path}",
        f"Results directory: {results_dir}",
        "",
        "=== Test Set Performance ===",
        f"Accuracy: {basic_accuracy:.4f}",
        "",
        "=== Classification Report ===",
        classification_rep,
        "",
        "=== Cross-Validation (5-fold) ===",
        f"Mean accuracy: {cv_mean:.4f}",
        f"Std deviation: {cv_std:.4f}",
        "",
        "=== Model Comparison (Basic vs Full Features) ===",
        comparison_text,
        "",
        "=== Edge Device Suitability ===",
        edge_suitability_text,
        "",
        f"Model file: {model_path.name}",
    ]
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Text report saved to: {report_path}")

    print("\n=== Summary ===")
    print("Basic features classifier ready for deployment.")
    print(f"Performance: {basic_accuracy:.4f} accuracy with only 3 input features.")
    print("Edge deployment: Optimised for speed and low computational overhead.")


if __name__ == "__main__":
    main()
