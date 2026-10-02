"""The README and the report page are rendered from results/; these keep them in sync."""

import json
import re

from uplift_targeting import config, report


def test_readme_is_the_rendered_template():
    readme = (config.ROOT / "README.md").read_text()
    assert readme == report.render_readme(), "README.md is stale: run `make report`."


def test_report_page_is_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SITE", tmp_path / "site")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    page, readme = report.run()
    html = page.read_text()
    assert not re.findall(r"\{\{[A-Z0-9_]+\}\}", html)
    data = re.search(r"const DATA = (\{.*?\});\n", html).group(1)
    payload = json.loads(data)
    assert {"spendCurves", "byCost", "criteo", "segments"} <= set(payload)
    assert readme.read_text() == report.render_readme()
