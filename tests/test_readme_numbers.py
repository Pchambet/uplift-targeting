"""The README quotes measured numbers; this keeps it in sync with ``results/``."""

from uplift_targeting import config
from uplift_targeting.narrative import HEADLINE_KEYS, load_numbers


def test_every_headline_number_in_readme_matches_results():
    readme = (config.ROOT / "README.md").read_text()
    numbers = load_numbers().text
    missing = {k: numbers[k] for k in HEADLINE_KEYS if numbers[k] not in readme}
    assert not missing, f"README is stale for: {missing}"
