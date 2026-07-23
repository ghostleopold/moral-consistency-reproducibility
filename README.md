# MoralConsistency

A Julia implementation for computing cooperation and stability indices of moral
systems in game-theoretic settings, together with the Python scripts that turn
its output into the figures of the accompanying paper.

The stability indices are estimated by Monte Carlo integration over a fixed set
of payoff configurations (1000 per game class), drawn once and archived in
`payoff_values.csv` so that every system and every error-rate combination is
compared against the same payoffs. That file ships with this repository.

## Interpreting the syntax

Players have behavioral rules `p` and assessment rules `d`, represented in
either binary or integer form. The bit order of the norms (`d`):

    BBDD, BBDC, BBCD, BBCC
    BGDD, BGDC, BGCD, BGCC
    GBDD, GBDC, GBCD, GBCC
    GGDD, GGDC, GGCD, GGCC

and of the action rules (`p`):

    BB, BG, GB, GG

(Heuristic: B and D -> 0, C and G -> 1, stepping through binary 0000 to 1111.)

## Requirements

**Julia** 1.10.10 -- the pinned `Manifest.toml` is resolved for this version.
Dependencies are listed in `Project.toml` and pinned in `Manifest.toml`; install
with `Pkg.instantiate()` (below).

**Python 3** (3.11 or later), for the figures. The scripts use:

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scipy`
- `tqdm`

Install with `pip install pandas numpy matplotlib seaborn scipy tqdm`.

**ImageMagick** (the `magick` command), for stitching the two-panel composite
figures. On macOS: `brew install imagemagick`; on Debian/Ubuntu:
`apt-get install imagemagick`. Without it, every standalone panel is still
produced; only the composites are skipped.

**Fonts.** The composite panels label their sub-panels with **Helvetica Neue
Bold**, and the plots themselves prefer **Helvetica** (falling back to Arial,
then DejaVu Sans). If these are not installed, ImageMagick and matplotlib
substitute a default face without error, so text may look slightly different
from the published figures. Install Helvetica Neue for an exact match.

## Installation

```sh
git clone https://github.com/juliangarcia/MoralConsistency.git
cd MoralConsistency
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```

## Data availability

The full strategy sweep (one row per moral system, for each error-rate
combination) is too large to host in git and is archived separately:

> **Data deposit:** _Moral consistency: strategy sweep (v1)_, Bridges (Monash).
> DOI: [10.26180/33068042](https://doi.org/10.26180/33068042).

Download and unzip the deposit, then put its two pieces where the scripts
expect them. From the repository root:

```sh
# 1. the per-error-rate sweep -> data/optimised_code_19.02.2025/
mkdir -p data
cp -R <deposit>/sweep data/optimised_code_19.02.2025

# 2. the derived set-membership table -> figures/venn_data.csv
cp <deposit>/derived/venn_data.csv figures/venn_data.csv
```

(`<deposit>` is the unzipped `moral-consistency-strategy-sweep_v1/` folder.) The
sweep folder name `optimised_code_19.02.2025` is the path the figure and
verification scripts read from; keep it exactly as written.

## Usage

### Recomputing the sweep (optional)

The deposit already contains the computed indices, so most users can skip this.
To regenerate them, run one configuration per error-rate combination:

```sh
julia --threads 6 main.jl --json data/optimised_code_19.02.2025/complete_chi_0.02_epsilon_0.02.json
```

Set `--threads` close to your physical core count. Evaluating the full strategy
space takes on the order of an hour per game class and error-rate combination on
a desktop machine. Each config's fields:

- `chi` — reputation assignment error probability
- `epsilon` — execution error probability
- `payoffs_filename` — the archived payoff samples (`payoff_values.csv`)
- `strategy_space` — `complete` (full space) or `toy` (quick test)
- `output_filename_local` / `output_filename_global` — output CSV paths

### Generating the figures

With the data in place (see above), from the `figures/` directory:

```sh
cd figures
python make_all_figures.py
```

This one script regenerates every figure the paper draws from this repository,
each written to the location its own script already uses:

| Figure | Output |
|---|---|
| Error-grid density panels (no jitter) | `density_scatter/density_panel_4x4_area_{False,True}.png` |
| Zoom scatter grid (jittered) | `scatter_plots/scatterplot_panel_4x4_zoom_area_False.png` |
| Disaggregated per-game density (jittered) | `density_scatter/density_{pd,sg,sh}_chi_0.02_epsilon_0.02.png` |
| 16 no-jitter SI composites | `panels_jitter_off_box_on/scatter_plot_panel_chi_*_area_False.png` |
| Main-text Fig. 2 source panel | `panels_jitter_on_box_on/scatter_plot_panel_chi_0.02_epsilon_0.02_area_False.png` |
| Area-weighted figure source panel | `panels_jitter_on_box_on_area/scatter_plot_panel_chi_0.02_epsilon_0.02_area_True.png` |

## Multi-threading

The index computation uses Julia threads with a thread-local caching strategy to
avoid contention on the equilibrium caches. Set `--threads` close to the number
of physical cores for best performance.

## License

MIT License.

## Authors

Developed by Jorge Pena, Julian Garcia, and Toby Handfield.
