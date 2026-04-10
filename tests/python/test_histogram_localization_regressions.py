# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for histogram localization bundles."""

import json
from pathlib import Path


def test_histogram_locale_bundles_define_all_new_metric_keys():
    project_root = Path(__file__).parent.parent.parent
    locale_dir = project_root / "src" / "visualizer" / "gui" / "resources" / "locales"
    required_metrics = {"position_x", "position_y", "position_z", "volume", "anisotropy", "erank"}

    for locale_path in sorted(locale_dir.glob("*.json")):
        data = json.loads(locale_path.read_text())
        metric_block = data["histogram"]["metric"]

        for metric_name in required_metrics:
            assert metric_name in metric_block, f"{locale_path.name} missing histogram.metric.{metric_name}"
            assert "label" in metric_block[metric_name], f"{locale_path.name} missing {metric_name}.label"
            assert "description" in metric_block[metric_name], f"{locale_path.name} missing {metric_name}.description"


def test_erank_locales_are_not_left_as_english_placeholders():
    project_root = Path(__file__).parent.parent.parent
    locale_dir = project_root / "src" / "visualizer" / "gui" / "resources" / "locales"

    en_metric = json.loads((locale_dir / "en.json").read_text())["histogram"]["metric"]["erank"]

    for locale_path in sorted(locale_dir.glob("*.json")):
        if locale_path.name == "en.json":
            continue

        metric = json.loads(locale_path.read_text())["histogram"]["metric"]["erank"]
        assert metric["label"] != en_metric["label"], f"{locale_path.name} erank label still uses English placeholder"
        assert metric["description"] != en_metric["description"], (
            f"{locale_path.name} erank description still uses English placeholder"
        )
