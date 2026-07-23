#!/usr/bin/env python3
"""
make_panels.py -- build the two-panel scatter composites for every error value,
with jitter and the grey dashed zoom-box each independently switchable.

    Left panel  (a): hexbin density   (plot_single_labeled_density)
    Right panel (b): zoomed scatter    (plot_single_zoom_jittered)

Two toggles, both default ON:

    --jitter / --no-jitter    positional noise on the scatter points (both panels)
    --box    / --no-box       grey dashed zoom-box drawn on panel a

Plus:

    --area                    area-weighted stability index (default: uniform)
    --out DIR                 output root (default names itself from the toggles)

Everything for one run lands under a single root so the four combinations never
tangle:

    <root>/density/   left panels
    <root>/zoom/      right panels
    <root>/           stitched a|b composites

Examples:
    ./make_panels.py                          # jitter on,  box on
    ./make_panels.py --no-jitter              # jitter off, box on
    ./make_panels.py --no-jitter --no-box     # jitter off, box off
    ./make_panels.py --jitter --no-box --out /tmp/preview
"""
import argparse
import os
import re
import subprocess
import sys

import scatterplot_large as sp

# --- stitching layout, copied verbatim from make_scatter_panel.sh -----------
FONT = "Helvetica-Neue-Bold"
FONTSIZE = 340
LABEL_X, LABEL_Y = 20, 10
GAP = 180
MARGIN_TOP = 390
MARGIN_SIDE = 40
FILL = "black"


def stitch(left, right, out):
    """Glue the two panels side by side and stamp the a/b labels."""
    left_w = int(subprocess.run(
        ["magick", "identify", "-format", "%w", left],
        check=True, capture_output=True, text=True).stdout)

    subprocess.run([
        "magick",
        "(", left, right, "+smush", str(GAP), ")",
        "-background", "white",
        "-gravity", "North", "-splice", f"0x{MARGIN_TOP}",
        "-gravity", "West", "-splice", f"{MARGIN_SIDE}x0",
        "-gravity", "East", "-splice", f"{MARGIN_SIDE}x0",
        "-gravity", "South", "-splice", f"0x{MARGIN_SIDE}",
        "-font", FONT, "-pointsize", str(FONTSIZE), "-fill", FILL,
        "-gravity", "NorthWest",
        "-annotate", f"+{LABEL_X}+{LABEL_Y}", "a",
        "-annotate", f"+{MARGIN_SIDE + left_w + GAP + LABEL_X}+{LABEL_Y}", "b",
        out,
    ], check=True)


# Restrict the figure-2 sweep to the manuscript's chosen error grid.
# Values are compared as floats so filename quirks (e.g. "0.0" vs "0.00")
# don't matter; the strings pulled off disk are passed through unchanged.
CHI_VALUES = {0.01, 0.02, 0.05, 0.1}
EPS_VALUES = {0.00, 0.01, 0.02, 0.1}


def error_combos(only=None):
    """(chi, eps) pairs for the selected global_complete CSVs on disk.

    only: an explicit (chi, eps) string tuple -> build just that pair,
          bypassing the CHI_VALUES x EPS_VALUES restriction. Used by the
          figure driver to build the single jittered composite that underlies
          main-text Fig. 2 without also rendering the fifteen unused panels.
    """
    if only is not None:
        return [only]
    combos = []
    for f in sorted(sp.files):
        m = re.search(r"chi_([0-9.]+)_epsilon_([0-9.]+)\.csv$", f)
        if m and float(m.group(1)) in CHI_VALUES and float(m.group(2)) in EPS_VALUES:
            combos.append((m.group(1), m.group(2)))
    return combos


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    jit = ap.add_mutually_exclusive_group()
    jit.add_argument('--jitter', dest='jitter', action='store_true',
                     help='add positional jitter (default)')
    jit.add_argument('--no-jitter', dest='jitter', action='store_false',
                     help='plot points at their exact values')
    ap.set_defaults(jitter=True)

    box = ap.add_mutually_exclusive_group()
    box.add_argument('--box', dest='box', action='store_true',
                     help='draw the grey dashed zoom-box on panel a (default)')
    box.add_argument('--no-box', dest='box', action='store_false',
                     help='omit the zoom-box')
    ap.set_defaults(box=True)

    ap.add_argument('--area', action='store_true',
                    help='use the area-weighted stability index')
    ap.add_argument('--out', default=None,
                    help='output root (default derived from the toggles)')
    ap.add_argument('--only', nargs=2, metavar=('CHI', 'EPS'), default=None,
                    help='build a single (chi, eps) pair, e.g. --only 0.02 0.02; '
                         'bypasses the 16-panel grid restriction')
    return ap.parse_args()


def main():
    args = parse_args()

    # Jitter amounts mirror the plotting defaults; zeroed when jitter is off.
    density_jitter = sp.JITTER_AMOUNT if args.jitter else 0.0
    zoom_jitter = sp.JITTER_AMOUNT / 5 if args.jitter else 0.0

    root = args.out or (
        f"panels_jitter_{'on' if args.jitter else 'off'}"
        f"_box_{'on' if args.box else 'off'}"
        + ("_area" if args.area else ""))
    density_dir = os.path.join(root, "density")
    zoom_dir = os.path.join(root, "zoom")
    os.makedirs(density_dir, exist_ok=True)
    os.makedirs(zoom_dir, exist_ok=True)

    print(f"jitter={'on' if args.jitter else 'off'}  "
          f"box={'on' if args.box else 'off'}  "
          f"area={args.area}  ->  {root}/")

    only = tuple(args.only) if args.only else None
    made, skipped = [], []
    for chi, eps in error_combos(only=only):
        tag = f"chi={chi}, eps={eps}"
        try:
            sp.plot_single_labeled_density(
                chi=chi, eps=eps, by_area=args.area,
                jitter_amount=density_jitter, draw_zoom_box=args.box,
                out_dir=density_dir,
            )
            sp.plot_single_zoom_jittered(
                chi=chi, eps=eps, by_area=args.area,
                jitter_amount=zoom_jitter, out_dir=zoom_dir, file_suffix="",
            )
        except FileNotFoundError as e:
            print(f"Skipping {tag}: {e}")
            skipped.append(tag)
            continue

        left = os.path.join(
            density_dir,
            f"scatterplot_density_chi_{chi}_epsilon_{eps}_area_{args.area}.png")
        right = os.path.join(
            zoom_dir,
            f"scatterplot_zoom_chi_{chi}_epsilon_{eps}_area_{args.area}.png")

        if not (os.path.isfile(left) and os.path.isfile(right)):
            # plot_single_zoom_jittered bows out silently when a parameter pair
            # has no consistent discriminators -- nothing to zoom into.
            print(f"Skipping composite for {tag}: a panel was not produced")
            skipped.append(tag)
            continue

        out = os.path.join(
            root,
            f"scatter_plot_panel_chi_{chi}_epsilon_{eps}_area_{args.area}.png")
        stitch(left, right, out)
        print(f"Wrote {out}")
        made.append(out)

    print(f"\n{len(made)} composites written to {root}/")
    if skipped:
        print(f"{len(skipped)} skipped: {', '.join(skipped)}")


if __name__ == "__main__":
    if not os.path.isdir(sp.folder):
        sys.exit(f"Data folder {sp.folder} not visible -- run from figures/.")
    main()
