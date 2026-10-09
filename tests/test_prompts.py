from typing import get_type_hints

from beam import PromptTemplate


def test_string_and_callable_prompts_work_on_supported_python_versions():
    hints = get_type_hints(PromptTemplate)
    assert "system" in hints and "user" in hints
    template = PromptTemplate(
        name="solver", system=lambda **kw: f"Solve {kw['task']}", user="Task: {task}"
    )
    assert template.render(task="2+2") == ("Solve 2+2", "Task: 2+2")
