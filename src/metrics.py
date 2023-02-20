def fidelity_plus(pred, pred_new):
    """Metric calculation to measure explainer
     robustness to graph noise.
        Arguments:
            pred: probabilities of original predictions
            pred_new: probabilites of new predictions

        Returns: metric int
        """
    total = 0

    for i in range(len(pred)):
        total += pred[i] - pred_new[i]

    fidelity = total / len(pred)

    return fidelity


def sparsity(size_explain, size_total):
    """Metric to measure sparsity of explainer methods
        Arguments:
            size_explain: size of important features/nodes of explainable method
            size_total: total size of original network

        Returns: metric int
        """
    pass