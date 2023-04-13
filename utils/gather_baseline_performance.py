import glob
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.plotting import fig2img


def text2img(text, figsize=(6, 2), dpi=200):
    fig = plt.Figure(figsize=figsize, dpi=dpi)  # same settings as the other plots
    ax = fig.add_subplot(111)
    ax.text(0.01, 0.05, text, transform=ax.transAxes, fontsize=32)
    ax.set_axis_off()
    fig.set_tight_layout(True)
    img = fig2img(fig)
    return img


def pil_grid(images, max_horiz=np.iinfo(int).max):
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new("RGB", (h_sizes[-1], v_sizes[-1]), color="white")
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    return im_grid


target_dir = Path("data/tmp/baseline_performance_comparison")
target_dir.mkdir(parents=True, exist_ok=True)

benchmark_name = "SynthFunctionBenchmark"

# get baseline folders
resultspath = Path("data/results/")
basename = "baselines_auto_*"
baseline_folders = glob.glob(str(resultspath / basename))

# for each baseline gather the plot filenames
plot_basefnames = [
    "comparison_static_boxplot.png",
    "comparison_static_boxplot_cumulativereward.png",
    "comparison_static_boxplot_cumulativereward_logregret.png",
    "comparison_static_boxplot_logregret.png",
]

n_folders = len(baseline_folders)
n_plots = len(plot_basefnames)

paths = {}
for baseline_folder in baseline_folders:
    synthfunname = baseline_folder.split("_")[-1]
    print(synthfunname)
    paths[synthfunname] = []
    for plot_basefname in plot_basefnames:
        old_path = Path(baseline_folder) / benchmark_name / plot_basefname
        new_path = Path(target_dir) / synthfunname
        new_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old_path, new_path / plot_basefname)
        paths[synthfunname].append(old_path)


images = []
for synthfunname, fnames in paths.items():
    images.append([Image.open(p) for p in fnames])
images = np.asanyarray(images, dtype=object).T

textimages = np.asanyarray([Image.fromarray(text2img(t)) for t in paths.keys()], dtype=object)
images = np.concatenate((textimages[None, :], images))

w_img, h_img = images[0, 0].size

n_cols = n_folders
image = pil_grid(images.flatten(), max_horiz=n_cols)
grid_fname = Path(target_dir) / "grid.png"
image.save(grid_fname)

img = np.array(image)
plt.imshow(img)
plt.axis("off")
plt.tight_layout()
plt.show()
