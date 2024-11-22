import pandas as pd
from matplotlib import pyplot as plt


def visualize_features(
    selector,
    k: int,
    feature_names: pd.Index,
    x_train: pd.DataFrame,
    y_target: pd.Series,
) -> None:
    """
    Visualize selected features and their correlation with the target.

    Args:
        selector: Feature selector object.
        k (int): Number of top features to visualize.
        feature_names (pd.Index): Index of feature names.
        x_train (pd.DataFrame): Training data.
        y_target (pd.Series): Target variable.

    Returns:
        None
    """
    # Get the indices of the selected features
    selected_indices = selector.get_support(indices=True)
    top_k_indices = selected_indices[:k]

    # Get feature scores (importances) or correlations with the target
    feature_scores = selector.estimator_.feature_importances_
    correlation_values = [x_train[feature].corr(pd.Series(y_target)) for feature in feature_names[top_k_indices]]

    # Create a bar chart to visualize feature scores and a line chart for correlations
    plt.figure(figsize=(12, 6))
    plt.bar(range(k), feature_scores[top_k_indices], label="Score", color="b", alpha=0.7)
    plt.plot(
        range(k),
        correlation_values,
        color="r",
        marker="o",
        linestyle="-",
        label="Correlation",
    )
    plt.xticks(range(k), [feature_names[i] for i in top_k_indices], rotation="vertical")
    plt.xlabel("Features")
    plt.ylabel("Score / Correlation")
    plt.title("Top {} Selected Features with Correlation".format(k))
    plt.legend()
    plt.show()
