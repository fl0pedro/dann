# from itertools import cycle
from matplotlib.ticker import PercentFormatter
from pathlib import Path
from tqdm import tqdm
import colorsys
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import re
import seaborn as sns
import subprocess

class SeabornFig2Grid():
  def __init__(self, seaborngrid, fig,  subplot_spec):
    self.fig = fig
    self.sg = seaborngrid
    self.subplot = subplot_spec
    if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
      isinstance(self.sg, sns.axisgrid.PairGrid):
      self._movegrid()
    elif isinstance(self.sg, sns.axisgrid.JointGrid):
      self._movejointgrid()
    self._finalize()

  def _movegrid(self):
    """ Move PairGrid or Facetgrid """
    self._resize()
    n = self.sg.axes.shape[0]
    m = self.sg.axes.shape[1]
    self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
    for i in range(n):
      for j in range(m):
        self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

  def _movejointgrid(self):
    """ Move Jointgrid """
    h= self.sg.ax_joint.get_position().height
    h2= self.sg.ax_marg_x.get_position().height
    r = int(np.round(h/h2))
    self._resize()
    self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

    self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
    self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
    self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

  def _moveaxes(self, ax, gs):
    #https://stackoverflow.com/a/46906599/4124317
    ax.remove()
    ax.figure=self.fig
    self.fig.axes.append(ax)
    self.fig.add_axes(ax)
    ax._subplotspec = gs
    ax.set_position(gs.get_position(self.fig))
    ax.set_subplotspec(gs)

  def _finalize(self):
    plt.close(self.sg.fig)
    self.fig.canvas.mpl_connect("resize_event", self._resize)
    self.fig.canvas.draw()

  def _resize(self, evt=None):
    self.sg.fig.set_size_inches(self.fig.get_size_inches())


filenames = subprocess.check_output(["bash", "-c", "find ../DATA -type f -name output*"]).decode().split('\n')

clean_up_rc = re.compile(r"(?:results_|_output|_all|_final|\.pkl)")
starts_with_rc = re.compile(r"^((?:[kfe]?mnist)|cifar10)_1_layer(?:_dropout|)$")

all_test_results = []

def multi_model_joint_plot(data, models, key, name, cols=4, scatter_kws=None):
  if scatter_kws is None:
    scatter_kws = {}
  rows = -(len(models)//-cols) or 1
  fig = plt.figure(figsize=(6.5*cols, 5*rows))
  gs = gridspec.GridSpec(rows, cols)

  num_bins = 20

  xmin = data["test_acc"].min()
  xmax = data["test_acc"].max()
  ymin = data["trainable_params"].min()
  ymax = data["trainable_params"].max()

  xbins = np.linspace(xmin, xmax, num_bins + 1)
  ybins = np.logspace(np.log10(ymin), np.log10(ymax), num_bins + 1)

  handles_labels = None
  
  filename = Path("figs", data["type"].iloc[0], name + ".png")

  # for i, ((model, data), color) in enumerate(zip(test_results.groupby("model"), cycle(sns.color_palette()[:5]))):
  # for i, (model, data) in enumerate(test_results.groupby("model")):
  # for i, ((model, data), color) in enumerate(zip(test_results.groupby("model"), (sns.color_palette("husl", 12)[(x*5)%12] for x in range(12)))):
  start = 0
  step = 31/11
  hues = np.arange(12)*step + start
  for i, ((k, data), color) in enumerate(zip(data.groupby(key), (colorsys.hls_to_rgb(hue, .60, .60) for hue in hues))):
    xpad = 0.05 * (xmax - xmin)
    ypad = 1.125

    # color = tuple(np.random.random((3,)).tolist())

    g = sns.JointGrid(
    data, x="test_acc", y="trainable_params", 
    space=0, height=6, ratio=5,
    xlim = (xmin - xpad, xmax + xpad),
    ylim=(ymin / ypad, ymax * ypad),
    marginal_ticks=False
  )

    sns.scatterplot(data=data, x="test_acc", y="trainable_params",
                  ax=g.ax_joint, color="black", alpha=0.5, s=10, **scatter_kws)
    sns.histplot(data=data, x="test_acc", bins=xbins, ax=g.ax_marg_x,
              color=color, alpha=0.25, linewidth=0.75)
    sns.histplot(data=data, y="trainable_params", bins=ybins, ax=g.ax_marg_y,
              color=color, alpha=0.25, linewidth=0.75)

    # g.ax_joint.set_xscale("log")
    g.ax_joint.set_yscale("log")
    # g.ax_marg_x.set_xscale("log")
    g.ax_marg_y.set_yscale("log")

    g.ax_joint.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    g.ax_marg_x.set_xlabel('')
    g.ax_marg_x.tick_params(labelbottom=False)

    g.ax_marg_x.set_title(k)

    xmean = data["test_acc"].mean()
    ymean = data["trainable_params"].mean()
    xstd = data["test_acc"].std()
    ystd = data["trainable_params"].std()
    xmax_local = data["test_acc"].max()
    ymax_local = data["trainable_params"].max()
  
    g.refline(x=xmean, y=ymean)

    g.ax_joint.text(
    x=xmin,
    y=ymax/ypad,
    s=
f"""n = {len(data)}
$\\mu_x$ = {xmean:.2%}
$\\sigma_x$ = {xstd:.2%}
$\\mu_y$ = {ymean:.1e}
$\\sigma_y$ = {ystd:.1e}
max$_x$ = {xmax_local:.2%}
max$_y$ = {ymax_local:.1e})""",
    rotation=0,
    ha='left',
    va='top',
    bbox=dict(
      boxstyle="round,pad=0.3", 
      fc="white", 
      ec="black", 
      alpha=0.7
    )
  )

    g.ax_joint.scatter(x=data["test_acc"].iloc[0], y=data["trainable_params"].iloc[0], color="black", s=75, linewidth=3, alpha=.5, marker="x")
    if g.ax_joint.get_legend() is not None:
      if handles_labels is None:
        handles_labels = g.ax_joint.get_legend_handles_labels()
      g.ax_joint.get_legend().remove()

    SeabornFig2Grid(g, fig, gs[i])

  fig.suptitle(name, fontsize=24, y=1.025)
  if handles_labels is not None:
    fig.legend(*handles_labels, loc="lower right")
  
  gs.tight_layout(fig)
  
  fig.savefig(filename, dpi=fig.dpi, bbox_inches='tight')
  del fig

pbar = tqdm(total=len(filenames) + 2, desc="generating figures") # (-1 + 3)
for filename in filenames[:-1]:
  with open(filename, 'rb') as f:
    results: dict[str, pd.DataFrame] = pickle.load(f)

  clean_name = clean_up_rc.sub('', '_'.join(filename.split('/')[2:]))

  match = starts_with_rc.match(clean_name)
  if match is None:
    typ = "other_fmnist"
    dataset = "fmnist"
    clean_name.replace("_fmnist", "")
  else:
    if clean_name.endswith("_dropout"):
      typ = "dropout"
    else:
      typ = "no_drop"
    dataset = match.group(1)
    clean_name.replace("_1_layer", "")

  test_results: pd.DataFrame = results["testing"]
  test_results["acc_to_params_ratio"] = test_results["test_acc"]/test_results["trainable_params"]
  test_results["file_name"] = clean_name + ".png"
  test_results["type"] = typ
  test_results["dataset"] = dataset
  test_results.sort_values("acc_to_params_ratio", ascending=False, inplace=True)

  all_test_results.append(test_results)
  
  multi_model_joint_plot(test_results, test_results["model"].unique(), "model", clean_name)
  pbar.update()

all_test_results = pd.concat(all_test_results, ignore_index=True)
all_test_results.sort_values("acc_to_params_ratio", ascending=False, inplace=True)

pbar.desc = "generating last figures"
for typ, data in all_test_results.groupby("type"):
  multi_model_joint_plot(
    data,
    all_test_results["dataset"].unique(),
    "dataset",
    "all_" + typ,
    cols = 3,
    scatter_kws=dict(hue="model")
  )
  pbar.update()

pickle.dump(all_test_results, open("all_test_results.pkl", "wb"))