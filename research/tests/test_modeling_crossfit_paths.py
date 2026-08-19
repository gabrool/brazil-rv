from pathlib import Path

from brazil_rv.modeling.crossfit import _extension_path


def test_extension_artifact_key_includes_fold_and_seed() -> None:
    root = Path("extensions")

    fold_a = _extension_path(root, Path("campaign/fold_a/seed_11"))
    fold_b = _extension_path(root, Path("campaign/fold_b/seed_11"))

    assert fold_a.name == "fold_a_seed_11.npz"
    assert fold_b.name == "fold_b_seed_11.npz"
    assert fold_a != fold_b

