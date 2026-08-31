from pathlib import Path
from jinja2 import Environment, FileSystemLoader

def get_system_prompt(task: str, fewshots: bool = True):
    HERE = Path(__file__).resolve().parent
    env = Environment(loader=FileSystemLoader(str(HERE / "templates")))

    template = env.get_template("base.j2")

    return template.render(task=task, fewshots=fewshots)


def get_test_system_prompt(task: str, fewshots: bool = True):
    HERE = Path(__file__).resolve().parent
    env = Environment(loader=FileSystemLoader(str(HERE / "test_templates")))

    template = env.get_template("base.j2")

    return template.render(task=task, fewshots=fewshots)