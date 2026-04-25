import os
from pathlib import Path

from src.evaluation.env import candidate_env_paths, load_evaluation_env


def test_candidate_env_paths_include_project_and_workspace_roots(tmp_path: Path) -> None:
    project_root = tmp_path / "kt-aicc" / "do-um"
    project_root.mkdir(parents=True)

    assert candidate_env_paths(project_root) == (
        project_root / ".env",
        project_root.parent / ".env",
    )


def test_load_evaluation_env_loads_parent_without_overriding_project(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "kt-aicc" / "do-um"
    project_root.mkdir(parents=True)
    (project_root / ".env").write_text(
        "EVAL_TEST_CHILD_ONLY=child\nEVAL_TEST_SHARED=child\n",
        encoding="utf-8",
    )
    (project_root.parent / ".env").write_text(
        "EVAL_TEST_PARENT_ONLY=parent\nEVAL_TEST_SHARED=parent\n",
        encoding="utf-8",
    )
    for name in [
        "EVAL_TEST_CHILD_ONLY",
        "EVAL_TEST_PARENT_ONLY",
        "EVAL_TEST_SHARED",
    ]:
        monkeypatch.delenv(name, raising=False)

    loaded = load_evaluation_env(project_root)

    assert loaded == [project_root / ".env", project_root.parent / ".env"]
    assert os.environ["EVAL_TEST_CHILD_ONLY"] == "child"
    assert os.environ["EVAL_TEST_PARENT_ONLY"] == "parent"
    assert os.environ["EVAL_TEST_SHARED"] == "child"
