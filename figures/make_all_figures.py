#!/usr/bin/env python3
"""
make_all_figures.py -- regenerate every figure the paper draws from this repo.

Run it from the figures/ directory, with the sweep data and venn_data.csv in
place (see the repository README) and ImageMagick on the PATH:

    ./make_all_figures.py

It writes each figure to the location its own script already uses -- this
driver only orchestrates, it does not relocate anything. The targets:

  density_scatter/
      density_panel_4x4_area_False.png     4x4 error grid, sigma-bar   (no jitter)
      density_panel_4x4_area_True.png      4x4 error grid, area-weighted (no jitter)

  scatter_plots/
      scatterplot_panel_4x4_zoom_area_False.png   4x4 zoom scatter (jittered)

  panels_jitter_off_box_on/
      scatter_plot_panel_chi_<chi>_epsilon_<eps>_area_False.png   16 SI panels
                                                                  (no jitter, box)

  panels_jitter_on_box_on/
      scatter_plot_panel_chi_0.02_epsilon_0.02_area_False.png     underlies
                                                                  main-text Fig. 2

  panels_jitter_on_box_on_area/
      scatter_plot_panel_chi_0.02_epsilon_0.02_area_True.png      underlies the
                                                                  area-weighted Fig.

  panels_by_game/
      scatter_plot_panel_chi_0.02_epsilon_0.02_{pd,sg,sh}.png     the three
                                                                  disaggregated
                                                                  a/b composites,
                                                                  y-axis sigma^PD
                                                                  / sigma^SG /
                                                                  sigma^SH

The two jitter-on composites are the a/b panels only: a grey zoom-box is drawn
on panel a, but NO connecting arrow. The final main-text and area-weighted
figures are hand-composed in the paper repository, where the arrow between the
box and panel b is placed and fine-tuned by hand. This driver deliberately
stops at the arrow-less PNG.

Steps that call ImageMagick (the composite panels) are skipped with a warning,
not a crash, if `magick` is not found -- so on a machine without it you still
get every standalone panel.
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = ("0.02", "0.02")  # chi, eps for Fig. 2 and the disaggregated panels

# Composite output roots, rebuilt from scratch on every run so a stale panel
# from an earlier exploratory sweep (e.g. an eps=0.05 column) can never survive
# alongside the paper's figures.
COMPOSITE_ROOTS = (
    "panels_jitter_off_box_on",
    "panels_jitter_on_box_on",
    "panels_jitter_on_box_on_area",
    "panels_by_game",
)

GAMES = ("PD", "SG", "SH")


def preflight():
    """Fail early, with a pointer to the README, if inputs are missing."""
    problems = []

    # scatterplot_large reads venn_data.csv from the working directory and the
    # sweep CSVs from ../data/optimised_code_19.02.2025/ (see sp.folder). Check
    # both before importing it, so a missing file is a clear message rather
    # than a stack trace on import.
    venn = os.path.join(HERE, "venn_data.csv")
    if not os.path.isfile(venn):
        problems.append(
            f"venn_data.csv not found in {HERE}. Download the data deposit and "
            "place derived/venn_data.csv here (see README, Data availability).")

    data_dir = os.path.normpath(os.path.join(HERE, "..", "data",
                                             "optimised_code_19.02.2025"))
    baseline_csv = os.path.join(
        data_dir, f"global_complete_chi_{BASELINE[0]}_epsilon_{BASELINE[1]}.csv")
    if not os.path.isfile(baseline_csv):
        problems.append(
            f"sweep data not found at {data_dir}/. Download the data deposit "
            "and place its sweep/ contents there (see README).")

    if problems:
        print("Cannot generate figures:\n")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    return shutil.which("magick") is not None


def run_panels(*args, label):
    """Invoke make_panels.py as a subprocess; return True on success."""
    cmd = [sys.executable, os.path.join(HERE, "make_panels.py"), *args]
    print(f"\n=== {label} ===\n$ {' '.join(cmd[1:])}")
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        print(f"  ! make_panels.py failed for: {label}")
        return False
    return True


def main():
    have_magick = preflight()
    os.chdir(HERE)  # sp uses relative paths; run from figures/

    # In-process matplotlib figures. Import here, after preflight, so a missing
    # venn_data.csv is reported cleanly rather than blowing up at import time.
    import scatterplot_large as sp

    print("=== 4x4 error-grid density panels (no jitter) ===")
    sp.plot_panel_4x4_density(by_area=False)   # density_scatter/..._area_False.png
    sp.plot_panel_4x4_density(by_area=True)    # density_scatter/..._area_True.png

    print("\n=== 4x4 zoom scatter (jittered) ===")
    sp.plot_panel_4x4_zoom(by_area=False)      # scatter_plots/scatterplot_panel_4x4_zoom_area_False.png

    composites_ok = True
    if not have_magick:
        print("\n! ImageMagick (`magick`) not found -- skipping the composite "
              "panels (SI 16-grid, Fig. 2 source, area-weighted source, "
              "per-game panels).")
        print("  Install ImageMagick and re-run to build them; every "
              "standalone panel above is already done.")
        composites_ok = False
    else:
        # Clear the composite roots so only freshly built paper panels remain.
        for root in COMPOSITE_ROOTS:
            path = os.path.join(HERE, root)
            if os.path.isdir(path):
                shutil.rmtree(path)
        # 16 no-jitter SI composites (box on by default).
        composites_ok &= run_panels(
            "--no-jitter", label="16 no-jitter SI composites -> panels_jitter_off_box_on/")
        # Jitter-on a/b panel underlying main-text Fig. 2 (baseline only).
        composites_ok &= run_panels(
            "--only", *BASELINE,
            label="Fig. 2 source composite -> panels_jitter_on_box_on/")
        # Jitter-on a/b panel underlying the area-weighted figure (baseline only).
        composites_ok &= run_panels(
            "--area", "--only", *BASELINE,
            label="area-weighted source composite -> panels_jitter_on_box_on_area/")
        # Per-game a/b composites (baseline errors), one per game, replacing the
        # old standalone single-game density panels.
        for game in GAMES:
            composites_ok &= run_panels(
                "--game", game, "--only", *BASELINE,
                label=f"{game} disaggregated composite -> panels_by_game/")

    print("\n" + "=" * 60)
    if composites_ok:
        print("All figures generated.")
    else:
        print("Standalone panels generated; some composites were skipped "
              "(see warnings above).")
    print("Reminder: main-text Fig. 2 and its area-weighted counterpart are "
          "hand-composed from the jitter-on PNGs; the connecting arrow is added "
          "by hand in the paper repository.")


if __name__ == "__main__":
    main()
