from __future__ import annotations

from tsqbev.gap_analysis import analyze_reset_gap


def test_gap_analysis_targets_dense_bev_reset() -> None:
    report = analyze_reset_gap()

    assert report.current_runtime == "legacy_sparse_query_multimodal_student"
    assert report.target_runtime == "dense-bev-multitask-reset"
    assert len(report.gaps) >= 5


def test_gap_analysis_marks_representation_and_detection_as_critical() -> None:
    report = analyze_reset_gap()
    severities = {gap.area: gap.severity for gap in report.gaps}

    assert severities["primary_representation"] == "critical"
    assert severities["detection_head"] == "critical"
