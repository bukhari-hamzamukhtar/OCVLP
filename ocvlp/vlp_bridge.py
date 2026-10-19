"""Import VLP's own code (the unmodified clone in ../vlp) so every condition uses
the same program search, grammar construction and PCFG weighting as the paper.

Two things have to be handled before VLP can be imported on this laptop:

1. VLP's type_system.py re-launches Python with os.execv when PYTHONHASHSEED is
   not set. That makes grammar hashing deterministic, but on Windows the
   re-launch detaches the process from the terminal. We therefore require the
   variable to be set by the caller and fail early with a clear message.
2. VLP's run_experiment.py imports `ray` at module level for its parallel
   search variants. We only use single-process heap search, so a small stand-in
   module is enough; nothing in the heap-search path calls ray.
"""

import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VLP_DIR = ROOT / "vlp"

if os.environ.get("PYTHONHASHSEED") != "0":
    raise RuntimeError(
        "Set PYTHONHASHSEED=0 before running (VLP needs deterministic hashing). "
        "Example: PYTHONHASHSEED=0 python -m ocvlp.experiment ..."
    )


def _install_ray_stub():
    if "ray" in sys.modules:
        return

    def remote(*args, **kwargs):
        # Used both as @ray.remote and as @ray.remote(num_cpus=1).
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

    ray = types.ModuleType("ray")
    ray.remote = remote
    ray.get = lambda x, *a, **k: x
    ray.init = lambda *a, **k: None
    util = types.ModuleType("ray.util")
    queue = types.ModuleType("ray.util.queue")

    class Empty(Exception):
        pass

    class Queue(list):
        pass

    queue.Empty = Empty
    queue.Queue = Queue
    util.queue = queue
    ray.util = util
    sys.modules["ray"] = ray
    sys.modules["ray.util"] = util
    sys.modules["ray.util.queue"] = queue


_install_ray_stub()

if str(VLP_DIR) not in sys.path:
    sys.path.insert(0, str(VLP_DIR))

# Re-exported VLP pieces used by our runner.
from method.type_system import (  # noqa: E402
    Arrow,
    List,
    PrimitiveType,
    IMG,
    OBJECT,
    PROPERTY,
    ACTION,
    INT,
    BOOL,
    STRING,
)
from method.dsl import DSL  # noqa: E402
from method.run_experiment import run_algorithm  # noqa: E402
from method.experiment_helper import make_program_checker_with_accuracy  # noqa: E402

__all__ = [
    "ROOT",
    "VLP_DIR",
    "Arrow",
    "List",
    "PrimitiveType",
    "IMG",
    "OBJECT",
    "PROPERTY",
    "ACTION",
    "INT",
    "BOOL",
    "STRING",
    "DSL",
    "run_algorithm",
    "make_program_checker_with_accuracy",
]
