# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for histogram metric extraction."""

from types import SimpleNamespace

import pytest


class _ModelStub:
    def __init__(self, lf, means, scaling):
        self._means = lf.Tensor.from_numpy(means)
        self._scaling = lf.Tensor.from_numpy(scaling)

    def get_means(self):
        return self._means

    def get_scaling(self):
        return self._scaling

    def get_opacity(self):
        raise AssertionError("Opacity should not be requested in this test")


def _translation_matrix(tx: float, ty: float, tz: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, ty],
        [0.0, 0.0, 1.0, tz],
        [0.0, 0.0, 0.0, 1.0],
    ]


@pytest.fixture
def histogram_panel_module():
    from lfs_plugins import histogram_panel

    return histogram_panel


def test_histogram_metrics_include_positions_volume_anisotropy_and_erank(histogram_panel_module):
    metric_ids = {metric.id for metric in histogram_panel_module.METRICS}

    assert {"position_x", "position_y", "position_z", "volume", "anisotropy", "erank"} <= metric_ids


def test_histogram_position_metrics_use_world_space_means(histogram_panel_module, lf, numpy):
    panel = histogram_panel_module.HistogramPanel()
    model = _ModelStub(
        lf,
        numpy.array([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.5]], dtype=numpy.float32),
        numpy.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]], dtype=numpy.float32),
    )

    splat_type = getattr(getattr(lf, "NodeType", None), "SPLAT", None)
    if splat_type is None:
        splat_type = lf.scene.NodeType.SPLAT

    scene = SimpleNamespace(
        get_nodes=lambda: [
            SimpleNamespace(
                id=7,
                parent_id=-1,
                visible=True,
                type=splat_type,
                gaussian_count=2,
                world_transform=_translation_matrix(10.0, -3.0, 0.5),
            )
        ]
    )

    panel._metric_id = "position_x"
    numpy.testing.assert_allclose(
        panel._extract_metric_values(scene, model).cpu().numpy(),
        numpy.array([11.0, 8.0], dtype=numpy.float32),
    )

    panel._metric_id = "position_y"
    numpy.testing.assert_allclose(
        panel._extract_metric_values(scene, model).cpu().numpy(),
        numpy.array([-1.0, -2.5], dtype=numpy.float32),
    )

    panel._metric_id = "position_z"
    numpy.testing.assert_allclose(
        panel._extract_metric_values(scene, model).cpu().numpy(),
        numpy.array([3.5, 5.0], dtype=numpy.float32),
    )


def test_histogram_volume_anisotropy_and_erank_metrics_match_gaussian_scales(histogram_panel_module, lf, numpy):
    panel = histogram_panel_module.HistogramPanel()
    model = _ModelStub(
        lf,
        numpy.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=numpy.float32),
        numpy.array([[1.0, 1.0, 1.0], [1.0, 2.0, 4.0]], dtype=numpy.float32),
    )
    scene = SimpleNamespace(get_nodes=lambda: [])

    panel._metric_id = "volume"
    numpy.testing.assert_allclose(
        panel._extract_metric_values(scene, model).cpu().numpy(),
        numpy.array([4.0 * numpy.pi / 3.0, 32.0 * numpy.pi / 3.0], dtype=numpy.float32),
        rtol=1e-6,
    )

    panel._metric_id = "anisotropy"
    anisotropy = panel._extract_metric_values(scene, model).cpu().numpy()
    numpy.testing.assert_allclose(anisotropy, numpy.array([1.0, 4.0], dtype=numpy.float32), rtol=1e-6)
    assert panel._histogram_bounds(lf.Tensor.from_numpy(anisotropy)) == (1.0, 4.0)

    panel._metric_id = "erank"
    erank = panel._extract_metric_values(scene, model).cpu().numpy()
    numpy.testing.assert_allclose(erank, numpy.array([3.0, 1.9503675], dtype=numpy.float32), rtol=1e-6)
    assert panel._histogram_bounds(lf.Tensor.from_numpy(erank)) == (1.0, 3.0)
