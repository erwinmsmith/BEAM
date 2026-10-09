import re
from pathlib import Path

import pytest

from beam.integrations.langgraph import apply_beam_masks


def test_named_mask_preserves_only_selected_edges():
    edges = [("b", "c"), ("a", "c"), ("a", "b")]
    mask = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    assert apply_beam_masks(edges, mask, ["a", "b", "c"]) == [("b", "c"), ("a", "b")]


def test_indexed_and_empty_masks():
    assert apply_beam_masks([(0, 1), (1, 0)], [[0, 1], [0, 0]]) == [(0, 1)]
    assert apply_beam_masks([(0, 1)], [[0, 0], [0, 0]]) == []
    assert apply_beam_masks([], []) == []


@pytest.mark.parametrize(
    "edges,mask,order",
    [
        ([], [[0, 1]], None),
        ([], [[0, 0.5], [0, 0]], None),
        ([], [[0, 1], [0, 0]], ["a", "a"]),
        ([("a", "b")], [[0, 1], [0, 0]], None),
        ([(0, 2)], [[0, 1], [0, 0]], None),
        ([("a", "c")], [[0, 1], [0, 0]], ["a", "b"]),
    ],
)
def test_mask_rejects_ambiguous_or_invalid_coordinates(edges, mask, order):
    with pytest.raises(ValueError):
        apply_beam_masks(edges, mask, order)


@pytest.mark.parametrize(
    "page",
    [
        "examples/examples.md",
        "examples/prompts-examples.md",
        "examples/basic-examples.md",
        "models/strategies.md",
        "models/LangGraph.md",
    ],
)
def test_offline_documentation_examples(page):
    path = Path(__file__).resolve().parents[1] / "docs" / page
    blocks = re.findall(r"```python\n(.*?)```", path.read_text(), re.S)
    assert blocks, f"No executable examples in {page}"
    namespace = {"__name__": "__main__"}
    for block in blocks:
        exec(compile(block, str(path), "exec"), namespace)
