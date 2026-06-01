import shutil
import importlib.resources as pkg_resources
from pathlib import Path

def init():
    """Copy the default bseflow.yaml template to the current working directory."""
    dest = Path.cwd() / "bseflow.yaml"
    if dest.exists():
        print("bseflow.yaml already exists in this directory — not overwriting.")
        return
    try:
        src = pkg_resources.files("bseflow").joinpath("default_config.yaml")
        shutil.copy(str(src), dest)
    except AttributeError:
        with pkg_resources.open_text("bseflow", "default_config.yaml") as f:
            dest.write_text(f.read())
    print("Created bseflow.yaml in current directory. Edit it for your environment.")