"""The CLI explains a missing input instead of failing with a traceback."""

import pytest

from uplift_targeting import cli, config


def test_missing_scores_point_to_the_step_that_makes_them(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "DATA_INTERIM", tmp_path / "data" / "interim")
    with pytest.raises(SystemExit, match="run `uplift-targeting model` first"):
        cli.main(["evaluate"])
