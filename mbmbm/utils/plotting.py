from pathlib import Path
from typing import Dict

from matplotlib import pyplot as plt

from mbmbm import LOGGED_TIMES_SAVE_NAME, logger
from mbmbm.utils.logging import get_formatted_time


def plot_logged_times(time_dict: Dict, plot_dir: Path):
    """
    Plots a bar graph of logged times from a dictionary and saves the figure to a given directory.

    This function processes a dictionary of logged times, plotting a normalized representation
    of these times as a bar chart, with the x-axis labels rotated for better readability,
    and then saves the plot to a specified directory.

    Parameters:
    time_dict (Dict): A dictionary where the key is a label (usually a string) and the value
                      is a list with [0]=string representation, [1]=numerical dureation,
                      [2]=time of logging.
    plot_dir (Path): A pathlib.Path object representing the directory in which to save the
                     created plot image. It is expected that the Path exists.

    The function logs the plotting process, sorts the times in ascending order, normalizes
    the times to make them sum to 1, creates a bar graph, adds formatted time labels to each bar,
    adjusts the x-axis labels, and saves the figure using the LOGGED_TIMES_SAVE_NAME filename
    in the specified plot directory. The actual name of the saved file should be set outside this
    function and be available as 'LOGGED_TIMES_SAVE_NAME'.
    """

    logger.info("Plot times")
    values = []
    labels = []

    keys = [p[0] for p in sorted([[k, v[2]] for k, v in time_dict.items()], key=lambda x: x[1])]
    for key in keys:
        labels.append(key)
        values.append(time_dict[key][1])
    sum_val = sum(values)
    values = [v / sum_val for v in values]

    fig, ax = plt.subplots()
    bar_container = ax.bar(labels, values)
    ax.set(ylabel="time")
    ax.bar_label(bar_container, fmt=lambda x: get_formatted_time(x, with_linebreak=True))
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plot_dir / LOGGED_TIMES_SAVE_NAME)
