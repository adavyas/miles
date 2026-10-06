import os
import shlex
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


def test_package_import_defers_actor_and_preserves_existing_globals() -> None:
    run_import_script(
        BLOCK_ACTOR
        + """
import logging
logging.getLogger().setLevel(logging.DEBUG)
import miles.backends.fsdp_utils as fsdp
assert fsdp.__all__ == ['load_fsdp_args', 'FSDPTrainRayActor']
assert 'miles.backends.fsdp_utils.actor' not in sys.modules
assert logging.getLogger().level == logging.WARNING
assert fsdp._FSDP_AVAILABLE is True
assert fsdp._TORCH_MEMORY_SAVER_AVAILABLE is True
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
from miles.utils.function_registry import load_function
assert load_function('miles.backends.fsdp_utils.FSDPTrainRayActor') is actor
assert load_function('miles.backends.fsdp_utils.actor.FSDPTrainRayActor') is actor
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


def test_public_exports_are_discoverable_before_resolution() -> None:
    run_import_script(
        BLOCK_ACTOR
        + """
import miles.backends.fsdp_utils as fsdp
assert set(fsdp.__all__) <= set(dir(fsdp))
assert 'miles.backends.fsdp_utils.actor' not in sys.modules
"""
    )


def test_resolved_exports_are_cached_in_package_namespace() -> None:
    run_import_script(
        """
import sys
from types import ModuleType
fake_actor = ModuleType('miles.backends.fsdp_utils.actor')
fake_actor.FSDPTrainRayActor = type('FSDPTrainRayActor', (), {})
sys.modules[fake_actor.__name__] = fake_actor
import miles.backends.fsdp_utils as fsdp
actor = fsdp.FSDPTrainRayActor
loader = fsdp.load_fsdp_args
assert vars(fsdp)['FSDPTrainRayActor'] is actor
assert vars(fsdp)['load_fsdp_args'] is loader
"""
    )


def test_rocm_build_smoke_check_requires_training_actor() -> None:
    dockerfile = (REPO_ROOT / "docker/Dockerfile.rocm").read_text()
    line = next(line for line in dockerfile.splitlines() if "miles backends: OK" in line)
    command = shlex.split(line.removeprefix("RUN "))
    assert command[:2] == ["python", "-c"]
    result = subprocess.run(
        [sys.executable, "-c", BLOCK_ACTOR + command[2]],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "training runtime unavailable" in result.stderr


def test_adaptation_registries_initialize_without_actor_import() -> None:
    run_import_script(
        BLOCK_ACTOR
        + """
from miles.backends.fsdp_utils.adaptations.weight_bridge import _REGISTRY
from miles.backends.fsdp_utils.adaptations.class_patches import _MODEL_PATCH_HOOKS
from miles.backends.fsdp_utils.adaptations.precision import _PRECISION_POLICY_HOOKS
assert {'qwen3_moe', 'glm4_moe_lite'} <= set(_REGISTRY)
assert {'fp8_checkpoint_guard', 'nemotron_h_pattern_repair', 'qwen3_moe_moe_patch'} <= {
    hook.name for hook in _MODEL_PATCH_HOOKS
}
assert 'qwen3_dense_true_on_policy' in {hook.name for hook in _PRECISION_POLICY_HOOKS}
assert 'miles.backends.fsdp_utils.actor' not in sys.modules
"""
    )
