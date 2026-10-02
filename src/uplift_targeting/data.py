"""Download, verify and shape the Hillstrom e-mail experiment.

The raw CSV is cached under ``data/raw`` and checked against a pinned SHA-256 so
that every number in the report is tied to one exact file.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from uplift_targeting import config

FEATURE_COLUMNS = (
    "recency",
    "history",
    "log_history",
    "mens",
    "womens",
    "newbie",
    "zip_urban",
    "zip_rural",
    "channel_web",
    "channel_multichannel",
    "history_tier",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, expected_sha256: str | None = None) -> Path:
    """Download ``url`` to ``dest`` once; later calls only verify the checksum."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "uplift-targeting"})
        with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
        # Verify before caching: a corrupted download must never become the cached file.
        if expected_sha256 is not None and sha256(tmp) != expected_sha256:
            tmp.unlink()
            raise ValueError(f"Checksum mismatch for {url}; the download was discarded.")
        tmp.replace(dest)
    elif expected_sha256 is not None and sha256(dest) != expected_sha256:
        raise ValueError(f"Checksum mismatch for {dest}; delete it and run `make data` again.")
    return dest


def hillstrom_path() -> Path:
    return config.DATA_RAW / "hillstrom.csv"


def fetch_hillstrom() -> Path:
    return download(config.HILLSTROM_URL, hillstrom_path(), config.HILLSTROM_SHA256)


@dataclass(frozen=True)
class Experiment:
    """Model-ready view of the experiment: features, arm, outcomes, raw columns."""

    X: pd.DataFrame
    arm: np.ndarray
    outcomes: pd.DataFrame
    raw: pd.DataFrame

    @property
    def n(self) -> int:
        return len(self.arm)


def purchase_history(mens: pd.Series, womens: pd.Series) -> pd.Series:
    """Collapse the two product-line flags into one readable segment."""
    labels = np.select(
        [(mens == 1) & (womens == 1), mens == 1, womens == 1],
        ["Both lines", "Mens only", "Womens only"],
        default="Neither",
    )
    return pd.Series(labels, index=mens.index, name="purchase_history")


def build_experiment(raw: pd.DataFrame) -> Experiment:
    """Encode covariates (all measured before randomisation) and outcomes.

    Only pre-treatment columns enter ``X``; ``visit``, ``conversion`` and
    ``spend`` are post-treatment and are kept apart so they cannot leak into a
    feature matrix by accident.
    """
    unknown = set(raw["segment"]) - set(config.ARMS)
    if unknown:
        raise ValueError(f"Unknown arms in data: {sorted(unknown)}")
    df = raw.copy()
    df["history_tier"] = df["history_segment"].str.slice(0, 1).astype(int)
    df["purchase_history"] = purchase_history(df["mens"], df["womens"])
    columns = {
        "recency": df["recency"].astype(float),
        "history": df["history"].astype(float),
        "log_history": np.log1p(df["history"].astype(float)),
        "mens": df["mens"].astype(float),
        "womens": df["womens"].astype(float),
        "newbie": df["newbie"].astype(float),
        "zip_urban": (df["zip_code"] == "Urban").astype(float),
        "zip_rural": (df["zip_code"] == "Rural").astype(float),
        "channel_web": (df["channel"] == "Web").astype(float),
        "channel_multichannel": (df["channel"] == "Multichannel").astype(float),
        "history_tier": df["history_tier"].astype(float),
    }
    X = pd.DataFrame({name: columns[name] for name in FEATURE_COLUMNS})
    arm = df["segment"].map(config.ARMS).to_numpy(dtype=np.int64)
    outcomes = df[list(config.OUTCOMES)].astype(float)
    return Experiment(X=X, arm=arm, outcomes=outcomes, raw=df)


def load_hillstrom(path: Path | None = None) -> Experiment:
    return build_experiment(pd.read_csv(path or hillstrom_path()))
