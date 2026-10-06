import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]

BLOCK_ACTOR = """
import importlib.abc
import sys

class BlockActor(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'miles.backends.fsdp_utils.actor':
            raise ImportError('training runtime unavailable')

sys.meta_path.insert(0, BlockActor())
"""


def run_import_script(script: str) -> None:
    # A fresh interpreter checks import ordering independently of pytest's
    # collected modules and any already initialized distributed runtime.
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_package_import_does_not_load_actor_or_change_root_logging() -> None:
    run_import_script(
        BLOCK_ACTOR
        + """
import logging
logging.getLogger().setLevel(logging.DEBUG)
import miles.backends.fsdp_utils as fsdp
assert fsdp.__all__ == ['load_fsdp_args', 'FSDPTrainRayActor']
assert 'miles.backends.fsdp_utils.actor' not in sys.modules
assert logging.getLogger().level == logging.DEBUG
"""
    )


def test_argument_and_scheduler_consumers_do_not_require_actor() -> None:
    run_import_script(
        BLOCK_ACTOR
        + """
from miles.backends.fsdp_utils import load_fsdp_args
from miles.backends.fsdp_utils.arguments import load_fsdp_args as direct
import miles.backends.fsdp_utils.lr_scheduler
assert load_fsdp_args is direct
assert 'miles.backends.fsdp_utils.actor' not in sys.modules
"""
    )


@pytest.mark.parametrize("style", ["attribute", "from_import", "star"])
def test_public_exports_resolve_to_original_objects(style: str) -> None:
    statements = {
        "attribute": "actor = fsdp.FSDPTrainRayActor; loader = fsdp.load_fsdp_args",
        "from_import": "from miles.backends.fsdp_utils import FSDPTrainRayActor as actor, load_fsdp_args as loader",
        "star": "from miles.backends.fsdp_utils import *; actor = FSDPTrainRayActor; loader = load_fsdp_args",
    }
    run_import_script(
        """
import sys
from types import ModuleType
fake_actor = ModuleType('miles.backends.fsdp_utils.actor')
fake_actor.FSDPTrainRayActor = type('FSDPTrainRayActor', (), {})
sys.modules[fake_actor.__name__] = fake_actor
import miles.backends.fsdp_utils as fsdp
"""
        + statements[style]
        + """
from miles.backends.fsdp_utils.arguments import load_fsdp_args as direct
assert actor is fake_actor.FSDPTrainRayActor
assert loader is direct
assert fsdp.FSDPTrainRayActor is actor
assert fsdp.load_fsdp_args is loader
"""
    )


def test_actor_dependency_error_is_deferred_and_preserved() -> None:
    run_import_script(
        BLOCK_ACTOR
        + """
import miles.backends.fsdp_utils as fsdp
try:
    fsdp.FSDPTrainRayActor
except ImportError as error:
    assert str(error) == 'training runtime unavailable'
else:
    raise AssertionError('missing actor dependency was swallowed')
"""
    )


def test_unknown_export_raises_attribute_error() -> None:
    run_import_script(
        BLOCK_ACTOR
        + """
import miles.backends.fsdp_utils as fsdp
try:
    fsdp.unknown_export
except AttributeError as error:
    assert 'unknown_export' in str(error)
else:
    raise AssertionError('unknown export did not raise AttributeError')
"""
    )
