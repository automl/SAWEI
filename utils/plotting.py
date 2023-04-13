import os
import typing
from typing import Any, List, Tuple

import matplotlib as mpl
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plottingscripts.utils.macros
import plottingscripts.utils.plot_util as plot_util
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.pyplot import figure, subplot, tick_params
from smac.facade.smac_ac_facade import SMAC4AC

from target_algorithms.synthetic_function_definitions import \
    synthetic_function_factory


def fig2img(fig, figsize=None, dpi=None):
    """Convert matplotlib figure to image as numpy array.

    :param fig: Plot to get image for.
    :type fig: matplotlib figure

    :param figsize: Optional figure size in inches, e.g. ``(10, 7)``.
    :type figsize: None or tuple of int

    :param dpi: Optional dpi.
    :type dpi: None or int

    :return: RGB image of plot
    :rtype: np.array
    """
    if dpi is not None:
        fig.set_dpi(dpi)
    if figsize is not None:
        fig.set_size_inches(figsize)
    # fig.set_tight_layout(True)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    image = np.reshape(image, (int(height), int(width), 3))

    # s, (width, height) = canvas.print_to_buffer()
    # image = np.fromstring(s, dtype=np.uint8).reshape((height, width, 3))

    return image


def plot_optimization_trace_mult_exp(
    time_list: typing.List,
    performance_list: typing.List,
    name_list: typing.List[str],
    title: str = None,
    logy: bool = False,
    logx: bool = False,
    properties: typing.Mapping = None,
    y_min: float = None,
    y_max: float = None,
    x_min: float = None,
    x_max: float = None,
    ylabel: str = "Performance",
    xlabel: str = "time [sec]",
    scale_std: float = 1,
    agglomeration: str = "mean",
    step: bool = False,
):
    """
    plot performance over time

    Arguments
    ---------
    time_list: typing.List[np.ndarray T]
        for each system (in name_list) T time stamps (on x)
    performance_list: typing.List[np.ndarray TxN]
        for each system (in name_list) an array of size T x N where N is the number of repeated runs of the system
    name_list: typing.List[str]
         names of all systems -- order has to be the same as in performance_list and time_list
    title: str
        title of the plot
    logy: bool
        y on log-scale
    logx: bool
        x on log-scale
    properties: typing.Mapping
        possible fields: "linestyles", "colors", "markers", "markersize", "labelfontsize", "linewidth", "titlefontsize",
                         "gridcolor", "gridalpha", "dpi", "legendsize", "legendlocation", "ticklabelsize",
                         "drawstyle", "incheswidth", "inchesheight", "loweryloglimit"
        > To turn off the legend, set legendlocation='None'
    y_min:float
        y min value
    y_max:float
        y max value
    x_min:float
        x min value
    x_max:float
        x max value
    ylabel: str
        y label
    xlabel: str
        y label
    scale_std: float
        scale of std (only used with agglomeration=="mean")
    agglomeration: str
        aggreation over repeated runs (either mean or median)
    step: bool
        plot as step function (True) or with linear interpolation (False)
    """

    if scale_std != 1 and agglomeration == "median":
        raise ValueError("Can not scale_std when plotting median")

    # complete properties
    if properties is None:
        properties = dict()
    properties = plot_util.fill_with_defaults(properties)

    # print(properties)

    # Set up figure
    ratio = 5
    gs = matplotlib.gridspec.GridSpec(ratio, 1)
    fig = figure(1, dpi=int(properties["dpi"]))
    fig.set_size_inches(properties["incheswidth"], properties["inchesheight"])

    ax1 = subplot(gs[0:ratio, :])
    ax1.grid(True, linestyle="-", which="major", color=properties["gridcolor"], alpha=float(properties["gridalpha"]))

    if title is not None:
        fig.suptitle(title, fontsize=int(properties["titlefontsize"]))

    auto_y_min = 2**64
    auto_y_max = -plottingscripts.utils.macros.MAXINT
    auto_x_min = 2**64
    auto_x_max = -(2**64)

    for idx, performance in enumerate(performance_list):
        performance = np.array(performance)
        color = next(properties["colors"])
        marker = next(properties["markers"])
        linestyle = next(properties["linestyles"])
        name_list[idx] = name_list[idx].replace("_", " ")

        axis = -1
        if logx and time_list[idx][0] == 0:
            time_list[idx][0] = 10**-1
        # print("Plot %s" % agglomeration)
        if agglomeration == "mean":
            m = np.mean(performance, axis=axis)
            lower = m - np.std(performance, axis=axis) * scale_std
            upper = m + np.std(performance, axis=axis) * scale_std
        elif agglomeration == "meanstderr":
            m = np.mean(performance, axis=axis)
            lower = m - (np.std(performance, axis=axis) / np.sqrt(performance.shape[0]))
            upper = m + (np.std(performance, axis=axis) / np.sqrt(performance.shape[0]))
        elif agglomeration == "median":
            m = np.median(performance, axis=axis)
            lower = np.percentile(performance, axis=axis, q=25)
            upper = np.percentile(performance, axis=axis, q=75)
        else:
            raise ValueError("Unknown agglomeration: %s" % agglomeration)

        if logy:
            lower[lower < properties["loweryloglimit"]] = properties["loweryloglimit"]
            upper[upper < properties["loweryloglimit"]] = properties["loweryloglimit"]
            m[m < properties["loweryloglimit"]] = properties["loweryloglimit"]

        # Plot m and fill between lower and upper
        if scale_std >= 0 and len(performance) > 1:
            ax1.fill_between(
                time_list[idx], lower, upper, facecolor=color, alpha=0.3, edgecolor=color, step="post" if step else None
            )
        if step:
            ax1.step(
                time_list[idx],
                m,
                color=color,
                linewidth=int(properties["linewidth"]),
                linestyle=linestyle,
                marker=marker,
                markersize=int(properties["markersize"]),
                label=name_list[idx],
                where="post",
                **properties.get("plot_args", {}),
            )

        else:
            ax1.plot(
                time_list[idx],
                m,
                color=color,
                linewidth=int(properties["linewidth"]),
                linestyle=linestyle,
                marker=marker,
                markersize=int(properties["markersize"]),
                label=name_list[idx],
                drawstyle=properties["drawstyle"],
                **properties.get("plot_args", {}),
            )

        # find out show from for this time_list
        show_from = 0
        if x_min is not None:
            for t_idx, t in enumerate(time_list[idx]):
                if t > x_min:
                    show_from = t_idx
                    break

        auto_y_min = min(min(lower[show_from:]), auto_y_min)
        auto_y_max = max(max(upper[show_from:]), auto_y_max)

        auto_x_min = min(time_list[idx][0], auto_x_min)
        auto_x_max = max(time_list[idx][-1], auto_x_max)

    # Describe axes
    if logy:
        ax1.set_yscale("log")
        auto_y_min = max(0.1, auto_y_min)
    ax1.set_ylabel("%s" % ylabel, fontsize=properties["labelfontsize"])

    if logx:
        ax1.set_xscale("log")
        auto_x_min = max(0.1, auto_x_min)
    ax1.set_xlabel(xlabel, fontsize=properties["labelfontsize"])

    if properties["legendlocation"] != "None":
        leg = ax1.legend(
            loc=properties["legendlocation"],
            fancybox=True,
            prop={"size": int(properties["legendsize"])},
            **properties.get("legend_args", {}),
        )
        leg.get_frame().set_alpha(0.5)

    tick_params(axis="both", which="major", labelsize=properties["ticklabelsize"])

    # Set axes limits
    if y_max is None and y_min is not None:
        ax1.set_ylim([y_min, auto_y_max + 0.01 * abs(auto_y_max - y_min)])
    elif y_max is not None and y_min is None:
        ax1.set_ylim([auto_y_min - 0.01 * abs(auto_y_max - auto_y_min), y_max])
    elif y_max is not None and y_min is not None and y_max > y_min:
        ax1.set_ylim([y_min, y_max])
    else:
        ax1.set_ylim(
            [auto_y_min - 0.01 * abs(auto_y_max - auto_y_min), auto_y_max + 0.01 * abs(auto_y_max - auto_y_min)]
        )

    if x_max is None and x_min is not None:
        ax1.set_xlim([x_min - 0.1 * abs(x_min), auto_x_max + 0.1 * abs(auto_x_max)])
    elif x_max is not None and x_min is None:
        ax1.set_xlim([auto_x_min - 0.1 * abs(auto_x_min), x_max + 0.1 * abs(x_max)])
    elif x_max is not None and x_min is not None and x_max > x_min:
        ax1.set_xlim([x_min, x_max])
    else:
        ax1.set_xlim([auto_x_min, auto_x_max + 0.1 * abs(auto_x_min - auto_x_max)])

    return fig


def plot_incumbents_over_time(data: pd.DataFrame, figfname: str = ""):
    dpi = 300
    figsize = (6, 4)
    fig = plt.figure()
    fig.set_size_inches(figsize)
    fig.set_dpi(dpi)
    ax = fig.add_subplot(1, 1, 1)
    ax = sns.lineplot(data=data, x="ta_time_used", y="train_perf", ax=ax, marker="o", hue="exp_id")
    # ax.set_yscale("log")
    # ax.set_xlim(0, 600)
    ax.set_ylim(-0.005, 0.1)
    nseeds = len(data["seed"].unique())
    nouter = len(data["outer_resampling_iter"].unique())
    ax.set_title(f"Performance over Time ($n_{{seeds}}={nseeds}, n_{{outer}}={nouter}$)")
    ax.set_xlabel("target algorithm: used time [s]")
    ax.set_ylabel("training cost (1 - accuracy)")
    fig.set_tight_layout(True)
    if figfname:
        fig.savefig(figfname, bbox_inches="tight", dpi=600)

    return fig


def plot_costs(
    costs: pd.DataFrame,
    silent: bool = False,
    fig_filename: str = "",
    title: str = "",
    hue: Any = None,
    whichcost: List[str] = [],
    logy: bool = True,
):
    # def c_in_whichcost(c):
    #     is_in_whichcost = []
    #     for wc in whichcost:
    #         if wc in c:
    #             is_in_whichcost.append(True)
    #         else:
    #             is_in_whichcost.append(False)
    #     return np.any(is_in_whichcost)

    with sns.axes_style("whitegrid"):
        figclass = plt.Figure if silent else plt.figure
        fig = figclass(figsize=(6, 4), dpi=250)
        ax = fig.add_subplot(111)
        if logy:
            ax.set_yscale("log")
        id_vars = ["seed", "step", "init_design"]
        cost_names = [c for c in costs.columns if c not in id_vars]
        if whichcost:
            cost_names = [c for c in cost_names if c in whichcost]
        costs_melted = costs.melt(id_vars=id_vars, value_vars=cost_names)
        sns.lineplot(data=costs_melted, x="step", y="value", ax=ax, hue="variable")
        title = "Costs" if not title else title
        ax.set_title(title)
        fig.set_tight_layout(True)
        if not silent:
            plt.show()
        if fig_filename:
            fig.savefig(fig_filename, bbox_inches="tight")

    return fig, ax


def plot_costs_comparedim(
    costs: pd.DataFrame,
    silent: bool = False,
    fig_filename: str = "",
    title: str = "",
    hue: Any = None,
    whichcost: List[str] = [],
):
    dims = costs["target_algorithm_dim"].unique()
    colors = sns.color_palette("husl", len(dims))
    whichcost = ["cost"]

    with sns.axes_style("whitegrid"):
        figclass = plt.Figure if silent else plt.figure
        fig = figclass(figsize=(6, 4), dpi=250)
        ax = fig.add_subplot(111)
        ax.set_yscale("log")
        id_vars = ["seed", "step"]
        cost_names = [c for c in costs.columns if c not in id_vars]
        if whichcost:
            cost_names = [c for c in cost_names if c in whichcost]
        # costs_melted = costs.melt(id_vars=id_vars, value_vars=cost_names)
        vals = costs["cost"].copy()
        vals[vals <= 0] = 1e-10
        costs["cost"] = vals
        sns.lineplot(data=costs, x="step", y="cost", ax=ax, hue="target_algorithm_dim", palette="pastel")
        title = "Costs" if not title else title
        ax.set_title(title)
        ax.yaxis.grid(True, which="both")
    if not silent:
        plt.show()
    if fig_filename:
        fig.savefig(fig_filename, bbox_inches="tight")

    return fig, ax


def plot_searchedspace(smac: SMAC4AC, fname: str = "", silent: bool = False):
    with sns.axes_style("white"):
        out_dir = ""
        if fname:
            out_dir = os.path.dirname(fname)
            fname = os.path.basename(fname)
        fig, ax = plot_costevalpoints(smac, out_dir=out_dir, fname=fname, silent=silent, mark_incumbent=True)
    return fig, ax


def plot_costevalpoints(
    smac: SMAC4AC, out_dir: str = "", silent: bool = False, mark_incumbent: bool = True, fname: str = ""
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Plot the evaluated configs with their associated cost (2D or 3D).

    Parameters
    ----------
    smac
    out_dir
    silent
    mark_incumbent
    fname

    Returns
    -------
    Tuple[mpl.figure.Figure, mpl.axes.Axes]

    """
    rh = smac.solver.runhistory
    X, Y = smac.solver.epm_chooser.rh2EPM.transform(rh)

    fig, ax = None, None
    n_dim = X.shape[-1]
    if n_dim == 2:
        fig = plt.figure() if not silent else plt.Figure()
        fig.set_dpi(600)
        ax = fig.add_subplot(111)
        cmap = plt.cm.coolwarm

        if smac.scenario.function_name:
            if smac.scenario.function_name in synthetic_function_factory:
                pass
                # xl = np.amin(X)
                # xu = np.amax(X)

                # fig, ax = plot_contour2d(
                #     smac.scenario.function_name,
                #     fig=fig,
                #     ax=ax,
                #     plot_bound=(xl, xu),
                #     cmap = cmap
                # )

        if np.any(Y < 0):
            norm = matplotlib.colors.Normalize()
        else:
            norm = matplotlib.colors.LogNorm()

        scatter = ax.scatter(
            X[:, 0],
            X[:, 1],
            c=Y[:, 0],
            cmap=cmap,
            norm=norm,
        )

        if mark_incumbent:
            incumbent = smac.solver.incumbent
            cost_inc = rh.get_cost(incumbent)

            inc = incumbent.get_array()

            ax.scatter(
                [inc[0]],
                [inc[1]],
                c="black",
                marker="*",
            )
            textpos = inc + [0.01, 0.01]
            ax.text(*textpos, f"{cost_inc:.4f}")

        ax.set_title(f"Cost of Evaluated Points ({type(smac).__name__})")

        keys = smac.scenario.cs.get_hyperparameter_names()
        ax.set_xlabel(keys[0])
        ax.set_ylabel(keys[1])
        fig.colorbar(scatter)

        if not silent:
            plt.show()

        if out_dir:
            if not fname:
                fname = "cost_evaluated_points.svg"
            out_fn = os.path.join(out_dir, fname)
            fig.savefig(out_fn, bbox_inches="tight")

    return fig, ax
