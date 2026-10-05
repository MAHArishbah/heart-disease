"""The rule that keeps the refactor honest.

Notebooks import from src/, never the reverse. A notebook that *defines*
something the pipeline needs is a notebook you will be copy-pasting from
forever, so re-defining any moved symbol is a test failure, not a style note.
"""

import ast
import json

import pytest

from disease_pred.config import PROJECT_ROOT

NOTEBOOKS = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb"))

# Symbols that now live in src/ and must not be re-declared in a notebook.
MOVED = {
    "load_raw", "clean", "select_features", "split", "prepare",
    "apply_zero_rules", "build_preprocessor", "output_names",
    "build_pipeline", "build_search", "param_grid", "cost_threshold",
    "oof_predictions", "score", "load_artifacts",
}


def notebook_code(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return [("".join(c["source"]), i)
            for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]


def test_three_notebooks_exist():
    assert [p.name for p in NOTEBOOKS] == [
        "01-eda.ipynb", "02-diagnostics.ipynb", "03-shap.ipynb"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_code_cell_parses(path):
    for source, i in notebook_code(path):
        try:
            ast.parse(source)
        except SyntaxError as exc:
            pytest.fail(f"{path.name} cell {i}: {exc}")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_imports_from_src(path):
    joined = "\n".join(s for s, _ in notebook_code(path))
    assert "disease_pred" in joined, f"{path.name} never imports the package"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_does_not_redefine_moved_symbols(path):
    for source, i in notebook_code(path):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.name not in MOVED, (
                    f"{path.name} cell {i} redefines {node.name}, "
                    "which lives in src/")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_empty_code_cells(path):
    """Original cells 62, 85 and 89 were empty and were deleted."""
    for source, i in notebook_code(path):
        assert source.strip(), f"{path.name} cell {i} is empty"


def test_src_never_imports_a_notebook():
    """src/ may *mention* notebooks in prose; it may not depend on one."""
    for module in (PROJECT_ROOT / "src" / "disease_pred").glob("*.py"):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                root = name.split(".")[0]
                assert root not in {"nbformat", "nbclient", "notebooks", "IPython"}, (
                    f"{module.name} imports {name}")
