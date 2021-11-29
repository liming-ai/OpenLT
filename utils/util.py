import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# WARNING:
# There is no guarantee that it will work or be used on a model. Please do use it with caution unless you make sure everything is working.
use_fp16 = False

if use_fp16:
    from torch.cuda.amp import autocast
else:
    class Autocast(): # This is a dummy autocast class
        def __init__(self):
            pass
        def __enter__(self, *args, **kwargs):
            pass
        def __call__(self, arg=None):
            if arg is None:
                return self
            return arg
        def __exit__(self, *args, **kwargs):
            pass

    autocast = Autocast()

def rename_parallel_state_dict(state_dict):
    count = 0
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            v = state_dict.pop(k)
            renamed = k[7:]
            state_dict[renamed] = v
            count += 1
    if count > 0:
        print("Detected DataParallel: Renamed {} parameters".format(count))
    return count

def rename_classifier_state_dict(state_dict):
    count = 0
    for k in list(state_dict.keys()):
        if k.startswith('model.classifier.'):
            v = state_dict.pop(k)
            renamed = 'model.classifier.classifier.' + k[17:]
            state_dict[renamed] = v
            count += 1
    if count > 0:
        print("Detected new classifier: Renamed {} parameters".format(count))
    return count

def load_state_dict(model, state_dict, no_ignore=False):
    own_state = model.state_dict()
    count = 0
    for name, param in state_dict.items():
        if name not in own_state: # ignore
            print("Warning: {} ignored because it does not exist in state_dict".format(name))
            assert not no_ignore, "Ignoring param that does not exist in model's own state dict is not allowed."
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
        except RuntimeError as e:
            print("Error in copying parameter {}, source shape: {}, destination shape: {}".format(name, param.shape, own_state[name].shape))
            raise e
        count += 1
    if count != len(own_state):
        print("Warning: Model has {} parameters, copied {} from state dict".format(len(own_state), count))
    return count

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambbda'''
    if alpha > 0:
        lamb = np.random.beta(alpha, alpha)
    else:
        lamb = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lamb * x + (1 - lamb) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lamb


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if isinstance(value, tuple) and len(value) == 2:
            value, n = value
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


# The calibration code is modified from https://github.com/hollance/reliability-diagrams/blob/master/reliability_diagrams.py
def calibration(true_labels, pred_labels, confidences, num_bins=15):
    """Collects predictions into bins used to draw a reliability diagram.
    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins
    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    pred_labels and true_labels may contain numeric or string labels.
    For a multi-class model, the predicted label and confidence should be those
    of the highest scoring class.
    Returns a dictionary containing the following NumPy arrays:
        accuracies: the average accuracy for each bin
        confidences: the average confidence for each bin
        counts: the number of examples in each bin
        bins: the confidence thresholds for each bin
        avg_accuracy: the accuracy over the entire test set
        avg_confidence: the average confidence over the entire test set
        expected_calibration_error: a weighted average of all calibration gaps
        max_calibration_error: the largest calibration gap across all bins
    """
    assert(len(confidences) == len(pred_labels))
    assert(len(confidences) == len(true_labels))
    assert(num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=np.float)
    bin_confidences = np.zeros(num_bins, dtype=np.float)
    bin_counts = np.zeros(num_bins, dtype=np.int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)

    return { "accuracies": bin_accuracies,
             "confidences": bin_confidences,
             "gaps": gaps,
             "counts": bin_counts,
             "bins": bins,
             "avg_accuracy": avg_acc,
             "avg_confidence": avg_conf,
             "expected_calibration_error": ece,
             "max_calibration_error": mce }


def _reliability_diagram_subplot(ax, bin_data,
                                 draw_ece=True,
                                 draw_acc=True,
                                 draw_bin_importance=False,
                                 title="Reliability Diagram",
                                 xlabel="Confidence",
                                 ylabel="Accuracy"):
    """Draws a reliability diagram into a subplot."""
    accuracies = bin_data["accuracies"]
    confidences = bin_data["confidences"]
    counts = bin_data["counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size/2.0

    widths = bin_size
    alphas = 0.3
    min_count = np.min(counts)
    max_count = np.max(counts)
    normalized_counts = (counts - min_count) / (max_count - min_count)

    if draw_bin_importance == "alpha":
        alphas = 0.2 + 0.8 * normalized_counts
    elif draw_bin_importance == "width":
        widths = 0.1 * bin_size + 0.9 * bin_size*normalized_counts

    colors = np.zeros((len(counts), 4))
    colors[:, 0] = 240 / 255.
    colors[:, 1] = 60 / 255.
    colors[:, 2] = 60 / 255.
    colors[:, 3] = alphas

    gap_plt = ax.bar(positions, np.abs(accuracies - confidences),
                     bottom=np.minimum(accuracies, confidences), width=widths,
                     edgecolor='white', color='#0a437a', linewidth=1, label="Gap")

    acc_plt = ax.bar(positions, accuracies, bottom=0, width=widths,
                     edgecolor="white", color="#448ee4", alpha=1.0, linewidth=1,
                     label="Accuracy")

    ax.set_aspect("equal")
    ax.plot([0,1], [0,1], linestyle = "--", color="gray")

    # ax.add_patch(
    #     patches.Rectangle(
    #         (0.71, 0.01),  # (x,y)
    #         0.285,         # width
    #         0.1,           # height
    #         facecolor = 'white',
    #         alpha=0.8,
    #         fill=True,
    #         transform=ax.transAxes
    #     )
    # )
    ax.add_patch(
        patches.Rectangle(
            (0.58, 0.01),  # (x,y)
            0.405,         # width
            0.15,           # height
            facecolor = 'white',
            alpha=0.8,
            fill=True,
            transform=ax.transAxes
        )
    )

    # single image
    # if draw_ece:
    #     ece = (bin_data["expected_calibration_error"] * 100)
    #     ax.text(0.98, 0.02, "ECE=%3.2f%%" % ece, color="black",
    #             ha="right", va="bottom", transform=ax.transAxes)
    # if draw_acc:
    #     acc = (bin_data["avg_accuracy"] * 100)
    #     ax.text(0.98, 0.06, "ACC=%3.2f%%" % acc, color="black",
    #             ha="right", va="bottom", transform=ax.transAxes)

    if draw_ece:
        ece = (bin_data["expected_calibration_error"] * 100)
        if ece > 10:
            ax.text(0.98, 0.02, "ECE=%.1f%%" % ece, color="black",
                    ha="right", va="bottom", transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0.98, 0.02, "ECE=%.2f%%" % ece, color="black",
                    ha="right", va="bottom", transform=ax.transAxes, fontsize=12)
    if draw_acc:
        acc = (bin_data["avg_accuracy"] * 100)
        if acc > 10:
            ax.text(0.98, 0.08, "ACC=%.1f%%" % acc, color="black",
                    ha="right", va="bottom", transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0.98, 0.08, "ACC=%.2f%%" % acc, color="black",
                    ha="right", va="bottom", transform=ax.transAxes, fontsize=12)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    #ax.set_xticks(bins)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(handles=[gap_plt, acc_plt])


def _reliability_diagram_combined(bin_data,
                                  draw_ece, draw_acc,
                                  draw_bin_importance, draw_averages,
                                  title, figsize, dpi=1080, save_fig=False):
    """Draws a reliability diagram and confidence histogram using the output
    from calibration()."""

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, dpi=dpi)

    # plt.tight_layout()
    # plt.subplots_adjust(hspace=-0.1)

    _reliability_diagram_subplot(ax, bin_data, draw_ece, draw_acc, draw_bin_importance,
                                 title=title, xlabel="Confidence")

    # Draw the confidence histogram upside down.
    # orig_counts = bin_data["counts"]
    # bin_data["counts"] = -bin_data["counts"]
    # _confidence_histogram_subplot(ax[1], bin_data, draw_averages, title="")
    # bin_data["counts"] = orig_counts

    # Also negate the ticks for the upside-down histogram.
    # new_ticks = np.abs(ax[1].get_yticks()).astype(np.int)
    # ax[1].set_yticklabels(new_ticks)

    # plt.show()
    if save_fig:
        plt.savefig('{}.png'.format(title), dpi=dpi, bbox_inches='tight')


def reliability_diagram(true_labels, pred_labels, confidences, num_bins=10,
                        draw_ece=True, draw_acc=True, draw_bin_importance=False,
                        draw_averages=False, title="Reliability Diagram",
                        figsize=(6, 6), dpi=1080, save_fig=False):
    """Draws a reliability diagram and confidence histogram in a single plot.

    First, the model's predictions are divided up into bins based on their
    confidence scores.
    The reliability diagram shows the gap between average accuracy and average
    confidence in each bin. These are the red bars.
    The black line is the accuracy, the other end of the bar is the confidence.
    Ideally, there is no gap and the black line is on the dotted diagonal.
    In that case, the model is properly calibrated and we can interpret the
    confidence scores as probabilities.
    The confidence histogram visualizes how many examples are in each bin.
    This is useful for judging how much each bin contributes to the calibration
    error.
    The confidence histogram also shows the overall accuracy and confidence.
    The closer these two lines are together, the better the calibration.

    The ECE or Expected Calibration Error is a summary statistic that gives the
    difference in expectation between confidence and accuracy. In other words,
    it's a weighted average of the gaps across all bins. A lower ECE is better.
    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins
        draw_ece: whether to include the Expected Calibration Error
        draw_bin_importance: whether to represent how much each bin contributes
            to the total accuracy: False, "alpha", "widths"
        draw_averages: whether to draw the overall accuracy and confidence in
            the confidence histogram
        title: optional title for the plot
        figsize: setting for matplotlib; height is ignored
        dpi: setting for matplotlib
        save_fig: if True, save the matplotlib Figure object
    """
    bin_data = calibration(true_labels, pred_labels, confidences, num_bins)
    _reliability_diagram_combined(bin_data, draw_ece, draw_acc, draw_bin_importance,
                                  draw_averages, title, figsize=figsize,
                                  dpi=dpi, save_fig=save_fig)
    return bin_data


def reliability_diagrams(results, num_bins=10,
                         draw_ece=True, draw_acc=True, draw_bin_importance=False,
                         num_cols=4, dpi=1080, save_fig=False):
    """Draws reliability diagrams for one or more models.

    Arguments:
        results: dictionary where the key is the model name and the value is
            a dictionary containing the true labels, predicated labels, and
            confidences for this model
        num_bins: number of bins
        draw_ece: whether to include the Expected Calibration Error
        draw_bin_importance: whether to represent how much each bin contributes
            to the total accuracy: False, "alpha", "widths"
        num_cols: how wide to make the plot
        dpi: setting for matplotlib
        save_fig: if True, save the matplotlib Figure object
    """
    ncols = num_cols
    nrows = (len(results) + ncols - 1) // ncols
    figsize = (ncols * 4, nrows * 4)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=False,
                           figsize=figsize, dpi=dpi, constrained_layout=False)

    for i, (plot_name, data) in enumerate(results.items()):
        y_true = data["true_labels"]
        y_pred = data["pred_labels"]
        y_conf = data["confidences"]

        bin_data = calibration(y_true, y_pred, y_conf, num_bins)

        row = i // ncols
        col = i % ncols
        _reliability_diagram_subplot(ax[row, col] if nrows > 1 else ax[col],
                                     bin_data, draw_ece, draw_acc,
                                     draw_bin_importance,
                                     title="\n".join(plot_name.split()),
                                     xlabel="Confidence",
                                     ylabel="Accuracy")

    # for i in range(i + 1, nrows * ncols):
    #     row = i // ncols
    #     col = i % ncols

    #     if nrows > 1:
    #         ax[row, col].axis("off")
    #     else:
    #         ax[col].axis("off")

    # plt.show()
    # if save_fig:
    # plt.title('Confidence', loc='center')
    plt.savefig('{}.png'.format('ece'), dpi=dpi, bbox_inches='tight')