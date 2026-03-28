from __future__ import annotations

from tsqbev.train import resolve_nuscenes_splits


def test_resolve_nuscenes_splits_defaults_to_mini() -> None:
    assert resolve_nuscenes_splits("v1.0-mini", None, None) == ("mini_train", "mini_val")


def test_resolve_nuscenes_splits_defaults_to_trainval() -> None:
    assert resolve_nuscenes_splits("v1.0-trainval", None, None) == ("train", "val")


def test_resolve_nuscenes_splits_preserves_explicit_values() -> None:
    assert resolve_nuscenes_splits("v1.0-mini", "custom_train", "custom_val") == (
        "custom_train",
        "custom_val",
    )
