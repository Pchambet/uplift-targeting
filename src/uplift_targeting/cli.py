"""Command-line entry point: ``uv run uplift-targeting <step>``.

Steps are idempotent and communicate only through files (``data/``,
``results/``), so any step can be re-run alone.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

from uplift_targeting import config


def _data(args: argparse.Namespace) -> None:
    from uplift_targeting import criteo, data

    print(f"Hillstrom: {data.fetch_hillstrom()}")
    print(f"Criteo:    {criteo.fetch()}")


def _readout(_: argparse.Namespace) -> None:
    from uplift_targeting import data, readout

    print(json.dumps(readout.run(data.load_hillstrom()), indent=2))


def _model(_: argparse.Namespace) -> None:
    from uplift_targeting import data, modeling

    scores = modeling.run(data.load_hillstrom())
    print(f"Out-of-fold scores: {scores.shape[0]} customers x {scores.shape[1]} columns")


def _evaluate(_: argparse.Namespace) -> None:
    from uplift_targeting import evaluation, modeling

    print(json.dumps(evaluation.run(modeling.load_scores()), indent=2))


def _criteo(_: argparse.Namespace) -> None:
    from uplift_targeting import criteo

    print(json.dumps(criteo.run(), indent=2))


def _figures(_: argparse.Namespace) -> None:
    from uplift_targeting import figures

    for path in figures.run():
        print(path.relative_to(config.ROOT))


def _report(_: argparse.Namespace) -> None:
    from uplift_targeting import report

    print(report.run().relative_to(config.ROOT))


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
        "report": (_report, "build site/index.html"),
        "run": (_run, "readout, model, evaluate, criteo, figures"),
    }
    for name, (func, help_text) in commands.items():
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=func)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
