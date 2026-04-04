from __future__ import annotations

from tsqbev.reset_stack import component_by_key, recommended_reset_plan, upstream_registry


def test_upstream_registry_contains_core_reset_components() -> None:
    registry_keys = {component.key for component in upstream_registry()}

    assert "openpcdet-centerpoint-pointpillar" in registry_keys
    assert "sparse4dv3" in registry_keys
    assert "bevformer-v2" in registry_keys
    assert "mit-bevfusion" in registry_keys
    assert "dinov2" in registry_keys
    assert "dinov3" in registry_keys
    assert "maptrv2" in registry_keys
    assert "efficientvit" in registry_keys


def test_recommended_reset_plan_points_to_foundation_teacher_runtime() -> None:
    plan = recommended_reset_plan()

    assert plan.name == "foundation-teacher-perspective-sparse-reset"
    assert any(component.role == "camera_sparse_runtime" for component in plan.components)
    assert any(component.role == "camera_perspective_supervision" for component in plan.components)
    assert any(component.role == "vector_map_head" for component in plan.components)
    assert plan.grid.temporal_frames == 4


def test_component_lookup_returns_expected_role() -> None:
    component = component_by_key("mit-bevfusion")

    assert component.role == "control_multimodal_runtime"
    assert component.deployability == "high"
