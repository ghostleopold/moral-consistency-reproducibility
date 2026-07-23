#!/usr/bin/env python3
"""
stage_deposit.py -- assemble the Bridges data deposit from the working tree.

Builds a clean, normalised copy of the sweep output under a new directory,
leaving the working tree untouched. Dry-run by default: it audits and prints
what it would do, and changes nothing until you pass --apply.

    ./stage_deposit.py                          # audit only, no writes
    ./stage_deposit.py --apply                  # build the deposit
    ./stage_deposit.py --apply --checksums      # ...and sha256 every file
    ./stage_deposit.py --manifest-only          # re-describe an edited deposit

What it fixes
-------------
1. Local files are excluded. The local_complete_* / local_properties_complete_*
   family (one long-format row per system per game) carries no information the
   global files lack: its stability_index is bit-identical to the global files'
   stability_index_PD/SG/SH columns, and its sampled_points is the constant 1000
   the README already states once. Rather than deposit ~1.4 GB of a redundant
   reshape, the sweep keeps only the global files.

2. Schema drift. The global CSVs carry three different column sets:

       20 cols : core 9 + 10 strategy-intrinsic booleans + pareto_front
       10 cols : core 9 + pareto_front
        9 cols : core 9

   The 10 booleans (is_conditional ... is_leading_eight) are keyed on (p,d)
   and do not vary with chi or epsilon -- they are verbatim copies of columns
   already in universal_properties.csv, repeated 1,048,576 times in each of
   12 files. They are dropped here rather than back-filled: the deposit then
   has one authoritative source for strategy-intrinsic properties
   (derived/universal_properties.csv) instead of thirteen. Pass
   --keep-booleans to preserve them instead (back-fill is NOT implemented --
   that flag keeps the drift, it does not repair it).

3. Sweep script. LIST_full.sh covers only 8 of the 23 runs. A complete
   run_full_sweep.sh is regenerated from the config files actually present.

4. macOS detritus and orphan images are simply not copied.

What it will NOT do silently
----------------------------
pareto_front is parameter-dependent (it is a frontier over
stability_index_uniform x cooperation_index within each file), so it cannot
be recovered by a join and is not dropped by default. It is missing from
global_complete_chi_0.01_epsilon_0.1.csv. Either regenerate it with
generate_pareto_frontier.jl before depositing, or pass --drop-pareto to
remove it everywhere and document it as recomputable. Shipping 22-of-23 is
the one option this script will complain about but still allow.
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC_SWEEP = REPO / "data" / "optimised_code_19.02.2025"

DEFAULT_OUT = REPO.parent / "moral-consistency-strategy-sweep_v1"

# Core per-parameter columns: everything that actually depends on chi/epsilon.
CORE_GLOBAL = [
    "p", "d", "resident_reputation_eq", "cooperation_index",
    "stability_index_PD", "stability_index_SG", "stability_index_SH",
    "stability_index_uniform", "stability_index_area",
]

# Strategy-intrinsic booleans -- duplicated from universal_properties.csv.
REDUNDANT_GLOBAL = [
    "is_conditional", "is_self_approving", "is_self_disapproving",
    "is_self_cooperative", "is_discriminating",
    "is_conditional_self_cooperative", "is_righteous", "is_consistent",
    "is_consistent_discriminating", "is_leading_eight",
]

PARETO = "pareto_front"

PARAM_RE = re.compile(r"chi_([0-9.eE+-]+)_epsilon_([0-9.eE+-]+)\.(csv|json)$")

# Extra files pulled in from the repo root / figures dir.
DERIVED = [
    (REPO / "universal_properties.csv", "universal_properties.csv"),
    (REPO / "figures" / "venn_data.csv", "venn_data.csv"),
]

SUBDIRS = ("configs", "sweep", "derived")

# Never staged, and purged from the deposit if they turn up: Finder writes
# these whenever the directory is browsed.
DETRITUS = {".DS_Store"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def params(name):
    """('0.01', '0.02') from any *_chi_X_epsilon_Y.{csv,json} filename."""
    m = PARAM_RE.search(name)
    return (m.group(1), m.group(2)) if m else None


def sort_key(pair):
    return (float(pair[0]), float(pair[1]))


def header_of(path):
    with open(path, "r") as fh:
        return fh.readline().rstrip("\n").split(",")


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def clone(src, dst):
    """Copy, preferring an APFS clone so the staging tree costs no disk."""
    try:
        subprocess.run(["cp", "-c", str(src), str(dst)], check=True,
                       capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        shutil.copy2(src, dst)


def project(src, dst, keep):
    """Write dst containing only the named columns of src, in `keep` order.

    Assumes unquoted CSV -- true for every file here (plain numerics and
    lowercase booleans). Verified against the headers before use.
    """
    with open(src, "r") as fin:
        head = fin.readline().rstrip("\n").split(",")
        idx = [head.index(c) for c in keep]
        with open(dst, "w") as fout:
            fout.write(",".join(keep) + "\n")
            if idx == list(range(len(head))):
                shutil.copyfileobj(fin, fout)
                return
            take = idx.__getitem__
            for line in fin:
                f = line.rstrip("\n").split(",")
                fout.write(",".join([f[i] for i in idx]) + "\n")


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------

def audit(drop_pareto):
    """Inspect the source tree. Returns (globals, configs, problems).

    `drop_pareto` only affects reporting: an inconsistency that the run is
    about to resolve is not worth reporting as an outstanding problem.

    Local files are intentionally not staged (see the module docstring), so
    they are neither collected nor reported here.
    """
    problems = []

    if not SRC_SWEEP.is_dir():
        sys.exit(f"Source sweep directory not found: {SRC_SWEEP}")

    gl, cfg = {}, {}
    for p in sorted(SRC_SWEEP.iterdir()):
        pr = params(p.name)
        if pr is None:
            continue
        if p.name.startswith("global_complete_chi_"):
            gl[pr] = p
        elif p.name.startswith("complete_chi_"):
            cfg[pr] = p

    print(f"source: {SRC_SWEEP}")
    print(f"  configs {len(cfg)}   global {len(gl)}   "
          f"(local files present but not staged)\n")

    missing = set(cfg) - set(gl)
    if missing:
        problems.append(
            f"{len(missing)} config(s) with no global output: "
            + ", ".join(f"chi={c} eps={e}"
                        for c, e in sorted(missing, key=sort_key)))
    orphan = set(gl) - set(cfg)
    if orphan:
        problems.append(
            f"{len(orphan)} global output(s) with no config: "
            + ", ".join(f"chi={c} eps={e}"
                        for c, e in sorted(orphan, key=sort_key)))

    # global schema variants
    print("global schema variants:")
    variants = {}
    for pr, p in gl.items():
        variants.setdefault(tuple(header_of(p)), []).append(pr)
    for cols, prs in sorted(variants.items(), key=lambda kv: -len(kv[1])):
        extra = [c for c in cols if c not in CORE_GLOBAL]
        print(f"  {len(prs):2d} file(s), {len(cols):2d} cols  "
              f"core+{extra if extra else '(nothing)'}")
    no_pareto = sorted((pr for pr, p in gl.items()
                        if PARETO not in header_of(p)), key=sort_key)
    if drop_pareto:
        print(f"  --drop-pareto: {PARETO} will be removed from all "
              f"{len(gl)} file(s), yielding one uniform "
              f"{len(CORE_GLOBAL)}-column schema")
    elif no_pareto:
        problems.append(
            f"{PARETO} missing from {len(no_pareto)} of {len(gl)} global "
            "file(s): "
            + ", ".join(f"chi={c} eps={e}" for c, e in no_pareto)
            + "  -> regenerate with generate_pareto_frontier.jl, "
              "or pass --drop-pareto")

    skipped = [p for p in SRC_SWEEP.iterdir()
               if p.name == ".DS_Store" or p.name == "LIST_full.sh"]
    skipped += [p for p in (REPO / "data").iterdir()
                if p.name == ".DS_Store" or p.name.startswith("transparent_")]
    if skipped:
        print("\nwill not be copied:")
        for p in sorted(skipped):
            print(f"  {p.relative_to(REPO)}")

    print("\nextra files:")
    for src, name in DERIVED:
        if src.exists():
            print(f"  ok      {name}  ({human(src.stat().st_size)})")
        else:
            problems.append(f"missing, will be skipped: {src}")

    return gl, cfg, problems


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def purge_detritus(out):
    """Remove Finder droppings from the deposit. Returns what it removed."""
    gone = []
    for p in sorted(out.rglob("*")):
        if p.name in DETRITUS and p.is_file():
            p.unlink()
            gone.append(p.relative_to(out).as_posix())
    return gone


def write_manifest(out, manifest):
    man = out / "MANIFEST.csv"
    with open(man, "w") as fh:
        fh.write("path,bytes,sha256,note\n")
        for row in manifest:
            fh.write(",".join(str(x) for x in row) + "\n")
    checksummed = sum(1 for r in manifest if r[2])
    print(f"\nwrote MANIFEST.csv ({len(manifest)} entries"
          f"{f', {checksummed} with checksums' if checksummed else ''})")


def write_readme(out, drop_pareto):
    (out / "README.md").write_text(README_STUB.format(
        pareto_row=(
            "" if drop_pareto else
            "| `pareto_front` | on the frontier of stability_index_uniform"
            " x cooperation_index, within this file |\n"),
        pareto_note=(
            "\n`pareto_front` is not included. It is a frontier over\n"
            "`stability_index_uniform` x `cooperation_index` computed"
            " within each\nfile, and is regenerated from these files by"
            " `generate_pareto_frontier.jl`\nin the code repository.\n"
            if drop_pareto else "")))
    print("wrote README.md (stub -- fill in the TODOs)")


def remanifest(out, drop_pareto, checksums):
    """Rebuild MANIFEST.csv and README.md from what is actually on disk.

    For use after editing a staged deposit by hand. This trusts the tree: it
    re-describes it, it does not re-verify it against the source.
    """
    if not out.is_dir():
        sys.exit(f"No deposit at {out}")

    gone = purge_detritus(out)
    for g in gone:
        print(f"  removed {g}")

    manifest = []
    for p in sorted(out.rglob("*")):
        if not p.is_file() or p.name in ("MANIFEST.csv", "README.md"):
            continue
        rel = p.relative_to(out).as_posix()
        size = p.stat().st_size
        manifest.append((rel, size, sha256_file(p) if checksums else "",
                         "staged"))

    print(f"\n{out.name}/")
    for sub in SUBDIRS:
        n = sum(1 for r in manifest if r[0].startswith(sub + "/"))
        tot = sum(r[1] for r in manifest if r[0].startswith(sub + "/"))
        print(f"  {sub + '/':14s} {n:3d} files  {human(tot):>9}")
    loose = [r for r in manifest
             if not any(r[0].startswith(s + "/") for s in SUBDIRS)]
    for r in loose:
        print(f"  (loose)        {r[0]}  {human(r[1])}")

    write_manifest(out, manifest)
    write_readme(out, drop_pareto)
    print(f"\ndeposit total: {human(sum(r[1] for r in manifest))} "
          f"across {len(manifest)} files")


def build(gl, cfg, out, keep_booleans, drop_pareto, checksums):
    for sub in SUBDIRS:
        (out / sub).mkdir(parents=True, exist_ok=True)

    manifest = []

    def record(path, note):
        size = path.stat().st_size
        digest = sha256_file(path) if checksums else ""
        manifest.append((path.relative_to(out).as_posix(), size, digest, note))
        print(f"  {human(size):>9}  {path.relative_to(out)}  {note}")

    print("\nconfigs/")
    for pr in sorted(cfg, key=sort_key):
        dst = out / "configs" / cfg[pr].name
        clone(cfg[pr], dst)
        record(dst, "copied")

    sweep_sh = out / "configs" / "run_full_sweep.sh"
    sweep_sh.write_text(
        "#!/bin/sh\n"
        "# Full parameter sweep: one invocation per configuration.\n"
        "# Run from the repository root with the configs on the path given.\n"
        "# Adjust -t to the number of physical cores available.\n\n"
        + "".join(f"julia -t 7 main.jl --json configs/{cfg[pr].name}\n"
                  for pr in sorted(cfg, key=sort_key)))
    sweep_sh.chmod(0o755)
    record(sweep_sh, f"generated, {len(cfg)} runs")

    print("\nsweep/")
    for pr in sorted(gl, key=sort_key):
        src = gl[pr]
        head = header_of(src)
        keep = [c for c in head if c in CORE_GLOBAL]
        if keep_booleans:
            keep += [c for c in head if c in REDUNDANT_GLOBAL]
        if PARETO in head and not drop_pareto:
            keep.append(PARETO)

        dst = out / "sweep" / src.name
        if keep == head:
            clone(src, dst)
            note = "copied"
        else:
            project(src, dst, keep)
            dropped = [c for c in head if c not in keep]
            saved = src.stat().st_size - dst.stat().st_size
            note = f"dropped {len(dropped)} col(s), saved {human(saved)}"
        record(dst, note)

    print("\nderived/")
    for src, name in DERIVED:
        if not src.exists():
            print(f"  SKIPPED  {name} (not found)")
            continue
        dst = out / "derived" / name
        clone(src, dst)
        record(dst, "copied")

    write_manifest(out, manifest)
    write_readme(out, drop_pareto)

    total = sum(r[1] for r in manifest)
    print(f"\ndeposit total: {human(total)} across {len(manifest)} files")
    print(f"staged at: {out}")


README_STUB = """# Moral consistency: strategy sweep (v1)

TODO: one-paragraph description of the study, and the citation for the paper
this dataset accompanies.

Code: <URL of the reproducibility repository, and its commit hash>

## Layout

    configs/       one JSON per parameter combination, plus run_full_sweep.sh
    sweep/         solver output, one global_ file per parameter combination
    derived/       strategy-intrinsic properties and set memberships

## sweep/

The global files cover the full strategy space of 1,048,576 systems
(16 action rules x 65,536 assessment rules), one row each.

`global_complete_chi_<chi>_epsilon_<eps>.csv`

| column | meaning |
|---|---|
| `p` | action rule, integer 0-15 |
| `d` | assessment rule, integer 0-65535 |
| `resident_reputation_eq` | TODO |
| `cooperation_index` | TODO |
| `stability_index_PD` | TODO (Prisoner's Dilemma) |
| `stability_index_SG` | TODO (Snowdrift) |
| `stability_index_SH` | TODO (Stag Hunt) |
| `stability_index_uniform` | TODO |
| `stability_index_area` | TODO |
{pareto_row}
Each stability index is a Monte Carlo estimate over the same 1000 archived
payoff configurations per game class (see the code repository's
`payoff_values.csv`).
{pareto_note}
## derived/

`universal_properties.csv` -- strategy-intrinsic properties, keyed on
`(p, d)`. These do not depend on chi or epsilon, which is why they are stored
once here rather than repeated in every file under `sweep/`. Join on `(p, d)`.

`venn_data.csv` -- `universal_properties.csv` plus the set memberships
(`set_C`, `set_A`, `set_A1`, `set_K`, `set_R`-`set_R4`, `is_set_Mc`-`is_set_Mo`)
and the `label` column naming the leading-eight strategies. Produced by
`generate_venn_categories.jl`. Every figure script joins against this file.

## Binary encoding

Assessment-rule bits, most significant first:

    BBDD, BBDC, BBCD, BBCC
    BGDD, BGDC, BGCD, BGCC
    GBDD, GBDC, GBCD, GBCC
    GGDD, GGDC, GGCD, GGCC

Action-rule bits: `BB, BG, GB, GG`. In both, B and D map to 0, C and G to 1.

## Integrity

`MANIFEST.csv` lists every file with its size in bytes and, if generated with
checksums, its SHA-256.

## License

TODO
"""


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually build the deposit (default: audit only)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output directory (default: {DEFAULT_OUT})")
    ap.add_argument("--keep-booleans", action="store_true",
                    help="keep the redundant strategy-intrinsic booleans in "
                         "the global files (preserves the drift, does not "
                         "repair it)")
    ap.add_argument("--drop-pareto", action="store_true",
                    help="drop pareto_front from every global file")
    ap.add_argument("--checksums", action="store_true",
                    help="sha256 every staged file (slow: several GB)")
    ap.add_argument("--manifest-only", action="store_true",
                    help="rebuild MANIFEST.csv and README.md from an existing "
                         "deposit, without restaging it -- use after editing "
                         "the deposit by hand")
    args = ap.parse_args()

    if args.manifest_only:
        remanifest(args.out, args.drop_pareto, args.checksums)
        return

    gl, cfg, problems = audit(args.drop_pareto)

    if problems:
        print("\n" + "=" * 70)
        print("PROBLEMS")
        for p in problems:
            print(f"  ! {p}")
        print("=" * 70)

    if not args.apply:
        print("\nDry run. Nothing written. Re-run with --apply to build.")
        return

    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"\nRefusing to write into non-empty {args.out}\n"
                 f"Remove it or pass a different --out.")

    print(f"\nbuilding deposit at {args.out}")
    build(gl, cfg, args.out, args.keep_booleans, args.drop_pareto,
          args.checksums)

    if problems:
        print("\nNote: the problems listed above were NOT fixed by this run.")


if __name__ == "__main__":
    main()
