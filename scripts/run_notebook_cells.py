"""Run only selected cells of a notebook (instead of re-executing everything).

A fresh kernel is started. The --deps cells are executed first to rebuild the state the
target cells need (their saved outputs are NOT changed). Then the --run cells are executed
and their new outputs are written back into the notebook.

Usage:
    python scripts/run_notebook_cells.py src/01_data_exploration.ipynb --deps 1,3,5 --run 40,41
    python scripts/run_notebook_cells.py NOTEBOOK --list        # show cell indices + first line
"""
import argparse
import copy
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def parse_ids(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()] if s else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--deps", default="", help="comma-separated cell indices to run silently first")
    ap.add_argument("--run", default="", help="comma-separated cell indices to run and save")
    ap.add_argument("--list", action="store_true", help="list cells and exit")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    path = Path(args.notebook).resolve()
    nb = nbformat.read(path, as_version=4)

    if args.list:
        for i, c in enumerate(nb.cells):
            first = c.source.strip().splitlines()[0] if c.source.strip() else ""
            print(f"{i:3d} {c.cell_type:8s} {first[:90]}")
        return

    deps, run = parse_ids(args.deps), parse_ids(args.run)
    for i in deps + run:
        if nb.cells[i].cell_type != "code":
            raise SystemExit(f"cell {i} is not a code cell")

    client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}})
    with client.setup_kernel():
        for i in deps:
            print(f"dep  cell {i} ...", flush=True)
            client.execute_cell(copy.deepcopy(nb.cells[i]), i)      # copy -> saved outputs untouched
        for i in run:
            print(f"run  cell {i} ...", flush=True)
            client.execute_cell(nb.cells[i], i)                      # outputs written into nb
    nbformat.write(nb, path)
    print(f"saved {path.name}: ran {run}")


if __name__ == "__main__":
    main()
