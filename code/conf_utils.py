import numpy as np


def calibration_error(
    y_true, y_prob, norm="l1", n_bins=15, strategy="uniform", reduce_bias=False
):
    """Compute calibration error of a binary classifier.

    Across all items in a set of N predictions, the calibration error measures
    the aggregated difference between (1) the average predicted probabilities
    assigned to the positive class, and (2) the frequencies
    of the positive class in the actual outcome.

    The calibration error is only appropriate for binary categorical outcomes.

    Read more in the :ref:`User Guide <calibration>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True predictions of a binary classification task.

    y_prob : array-like of (n_samples,)
        Probabilities of the positive class.

    norm : {'l1', 'l2', 'max'}, default='l2'
        Norm method. The l1-norm is the Expected Calibration Error (ECE),
        and the max-norm corresponds to Maximum Calibration Error (MCE).

    n_bins : int, default=10
       The number of bins to compute error on.

    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy used to define the widths of the bins.

        uniform
            All bins have identical widths.
        quantile
            All bins have the same number of points.
    reduce_bias : bool, default=True
        Add debiasing term as in Verified Uncertainty Calibration, A. Kumar.
        Only effective for the l2-norm.

    Returns
    -------
    score : float
        calibration error
    """

    if any(y_prob < 0) or any(y_prob > 1):
        raise ValueError("y_prob has values outside of [0, 1] range")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            "Only binary classification is supported. " "Provided labels %s." % labels
        )

    norm_options = ("l1", "l2", "max")
    if norm not in norm_options:
        raise ValueError(f"norm has to be one of {norm_options}, got: {norm}.")

    remapping = np.argsort(y_prob)
    y_true = y_true[remapping]
    y_prob = y_prob[remapping]
    sample_weight = np.ones(y_true.shape[0])

    n_bins = int(n_bins)
    if strategy == "quantile":
        quantiles = np.percentile(y_prob, np.arange(0, 1, 1.0 / n_bins) * 100)
    elif strategy == "uniform":
        quantiles = np.arange(0, 1, 1.0 / n_bins)
    else:
        raise ValueError(
            f"Invalid entry to 'strategy' input. Strategy must be either "
            f"'quantile' or 'uniform'. Got {strategy} instead."
        )

    threshold_indices = np.searchsorted(y_prob, quantiles).tolist()
    threshold_indices.append(y_true.shape[0])
    avg_pred_true = np.zeros(n_bins)
    bin_centroid = np.zeros(n_bins)
    delta_count = np.zeros(n_bins)
    debias = np.zeros(n_bins)

    loss = 0.0
    count = float(sample_weight.sum())
    for i, i_start in enumerate(threshold_indices[:-1]):
        i_end = threshold_indices[i + 1]
        # ignore empty bins
        if i_end == i_start:
            continue
        delta_count[i] = float(sample_weight[i_start:i_end].sum())
        avg_pred_true[i] = (
            np.dot(y_true[i_start:i_end], sample_weight[i_start:i_end]) / delta_count[i]
        )
        bin_centroid[i] = (
            np.dot(y_prob[i_start:i_end], sample_weight[i_start:i_end]) / delta_count[i]
        )
        if norm == "l2" and reduce_bias:
            delta_debias = avg_pred_true[i] * (avg_pred_true[i] - 1) * delta_count[i]
            delta_debias /= count * delta_count[i] - 1
            debias[i] = delta_debias

    if norm == "max":
        loss = np.max(np.abs(avg_pred_true - bin_centroid))
    elif norm == "l1":
        delta_loss = np.abs(avg_pred_true - bin_centroid) * delta_count
        loss = np.sum(delta_loss) / count
    elif norm == "l2":
        delta_loss = (avg_pred_true - bin_centroid) ** 2 * delta_count
        loss = np.sum(delta_loss) / count
        if reduce_bias:
            loss += np.sum(debias)
        loss = np.sqrt(max(loss, 0.0))
    return loss


def expected_calibration_error(pred_correct, pred_conf, num_bins=15):
    """
    Compute the Expected Calibration Error (ECE) for a set of predictions.

    Parameters:
        pred_correct (array-like): Correct prediction mask.
        pred_conf (array-like): Confidence of the predictions.
        num_bins (int): Number of bins to use for calibration. Default is 15.

    Returns:
        float: Expected Calibration Error (ECE).
    """
    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(pred_conf, bins[1:-1])

    ece = 0.0
    for bin_idx in range(1, num_bins + 1):
        mask = bin_indices == bin_idx
        if np.sum(mask) > 0:
            conf_bin = np.mean(pred_conf[mask])
            acc_bin = np.mean(pred_correct[mask])
            ece += np.abs(acc_bin - conf_bin) * np.sum(mask) / len(pred_correct)

    return ece


def overconfidence(pred_correct, pred_conf):
    """
    Compute the Overconfidence for a set of predictions.

    Parameters:
        pred_correct (array-like): Correct prediction mask.
        pred_conf (array-like): Confidence of the predictions.

    Returns:
        float: Overconfidence.
    """

    return pred_conf[pred_correct == 0].mean()


def underconfidence(pred_correct, pred_conf):
    """
    Compute the Underconfidence for a set of predictions.

    Parameters:
        pred_correct (array-like): Correct prediction mask.
        pred_conf (array-like): Confidence of the predictions.

    Returns:
        float: Underconfidence.
    """

    return (1 - pred_conf[pred_correct == 1]).mean()
