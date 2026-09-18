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

**Operating systems.** The Julia pipeline and the demo below were tested on
Linux (Fedora 44, kernel 7.1, x86-64) with Julia 1.10.10; the figure scripts
were developed on macOS. Any platform supported by Julia 1.10 should work.

**Hardware.** No non-standard hardware is required. A multi-core desktop or
laptop is recommended for the full sweep (see Multi-threading); the demo below
runs on a single core.

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
git clone https://github.com/ghostleopold/moral-consistency-reproducibility.git
cd moral-consistency-reproducibility
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```

Typical install time on a normal desktop computer: about one minute for
`Pkg.instantiate()` once Julia is installed (45 s on a 2025 laptop, dominated
by precompilation), plus the package downloads.

## Demo

A small demo runs the complete pipeline on a toy strategy space against a
mini payoff sample, both shipped with the repository:

- `toy_input.json` -- the demo configuration (chi = 0.01, epsilon = 0.02,
  `strategy_space: toy`);
- `mini_payoffs.csv` -- 9 payoff configurations, 3 per game class (PD, SG,
  SH), in the same format as the archived `payoff_values.csv`.

The toy space pairs 4 action rules (`p` = 8 to 11) with 7 social norms
(`d` = 0, 100, 1000, 1001, 1002, 50124, 53196), 28 moral systems in total.
`main.jl` uses `toy_input.json` when no `--json` argument is given:

```sh
julia main.jl
```

Expected run time: under 10 seconds on a normal desktop computer (6.6 s
including Julia start-up on a 2025 laptop, single thread). Expected output: the
console prints `Running test file...` and a progress bar, and two CSV files are
written to the repository root:

- `output_local_toy_show.csv` (84 rows): one row per moral system and game
  class, with columns `p, d, game, stability_index, sampled_points`. For the
  demo, `sampled_points` is 3 for every row (the 3 payoff configurations of
  that class) and `stability_index` is the fraction of them at which the
  system is stable, so it takes the values 0, 1/3, 2/3, or 1.
- `output_global_toy_show.csv` (28 rows): one row per moral system, with
  columns `p, d, resident_reputation_eq, cooperation_index,
  stability_index_PD, stability_index_SG, stability_index_SH,
  stability_index_uniform, stability_index_area`. For example, the first row
  (`p` = 8, `d` = 0) has resident reputation equilibrium 0.01, cooperation
  index 9.8e-5, and stability indices (0, 0, 1) for (PD, SG, SH), giving a
  uniform average of 1/3.

These files are the same kind of output the full sweep produces (with
`sampled_points` = 1000 and 524,800 rows per game class), so the demo exercises
every step of `compute_local_properties` and `compute_global_properties`.

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
| Disaggregated per-game a/b composites (jittered) | `panels_by_game/scatter_plot_panel_chi_0.02_epsilon_0.02_{pd,sg,sh}.png` |
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
