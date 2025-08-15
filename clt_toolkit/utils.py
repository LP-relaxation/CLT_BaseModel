from pathlib import Path
from dataclasses import replace

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def updated_dataclass(original, updates):
    return replace(original, **updates)