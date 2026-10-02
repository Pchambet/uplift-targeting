"""Command-line entry point: ``uv run uplift-targeting <step>``.

Steps are idempotent and communicate only through files (``data/``,
``results/``), so any step can be re-run alone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from uplift_targeting import config

if TYPE_CHECKING:
    from uplift_targeting.data import Experiment


class MissingInput(RuntimeError):
    """A step's input file does not exist yet: an earlier step has to run first."""


def _require(path: Path, step: str) -> None:
    if not path.exists():
        raise MissingInput(f"{path.relative_to(config.ROOT)} is missing: run `{step}` first.")


def _data(_: argparse.Namespace) -> None:
    from uplift_targeting import criteo, data

    print(f"Hillstrom: {data.fetch_hillstrom()}")
    print(f"Criteo:    {criteo.fetch()}")


def _hillstrom() -> Experiment:
    from uplift_targeting import data

    _require(data.hillstrom_path(), "make data")
    return data.load_hillstrom()


def _readout(_: argparse.Namespace) -> None:
    from uplift_targeting import readout

    print(json.dumps(readout.run(_hillstrom()), indent=2))


def _model(_: argparse.Namespace) -> None:
    from uplift_targeting import modeling

    scores = modeling.run(_hillstrom())
    print(f"Out-of-fold scores: {scores.shape[0]} customers x {scores.shape[1]} columns")


def _evaluate(_: argparse.Namespace) -> None:
    from uplift_targeting import evaluation, modeling

    _require(modeling.scores_path(), "uplift-targeting model")
    print(json.dumps(evaluation.run(modeling.load_scores()), indent=2))


def _criteo(args: argparse.Namespace) -> None:
    from uplift_targeting import criteo

    _require(criteo.raw_path(), "make data")
    print(json.dumps(criteo.run(refit=args.refit), indent=2))


def _figures(_: argparse.Namespace) -> None:
    from uplift_targeting import figures

    _require(config.RESULTS / "criteo_summary.json", "make run")
    for path in figures.run():
        print(path.relative_to(config.ROOT))


def _report(_: argparse.Namespace) -> None:
    from uplift_targeting import report

    _require(config.RESULTS / "criteo_summary.json", "make run")
    for path in report.run():
        print(path.relative_to(config.ROOT))


def _run(args: argparse.Namespace) -> None:
    steps: list[Callable[[argparse.Namespace], None]] = [
        _readout,
        _model,
        _evaluate,
        _criteo,
        _figures,
    ]
    for step in steps:
        start = time.perf_counter()
        step(args)
        print(f"[{step.__name__.strip('_')}] done in {time.perf_counter() - start:.0f}s")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="uplift-targeting", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {
        "data": (_data, "download and verify the raw data"),
        "readout": (_readout, "Part A: experiment readout"),
        "model": (_model, "Part B: out-of-fold uplift and response scores"),
        "evaluate": (_evaluate, "Parts B-C: Qini, policy values, cost sensitivity"),
        "criteo": (_criteo, "Part D: the same models on 14M Criteo rows"),
        "figures": (_figures, "static figures for the README"),
        "report": (_report, "build site/index.html and render README.md"),
        "run": (_run, "readout, model, evaluate, criteo, figures"),
    }
    for name, (func, help_text) in commands.items():
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=func)
        if name in ("criteo", "run"):
            p.add_argument(
                "--refit", action="store_true", help="refit Criteo models even if cached"
            )
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except MissingInput as err:
        sys.exit(f"uplift-targeting: {err}")


if __name__ == "__main__":
    main()
