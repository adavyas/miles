import logging


try:
    _TORCH_MEMORY_SAVER_AVAILABLE = True
except ImportError:
    logging.warning("torch_memory_saver is not installed, refer to : https://github.com/fzyzcjy/torch_memory_saver")
    _TORCH_MEMORY_SAVER_AVAILABLE = False

try:
    _FSDP_AVAILABLE = True
except ImportError as e:
    logging.warning(f"FSDP backend dependencies not available: {e}")
    _FSDP_AVAILABLE = False

__all__ = ["load_fsdp_args", "FSDPTrainRayActor"]


def __getattr__(name):
    # Argument and tensor consumers do not need the Ray training runtime.
    if name == "FSDPTrainRayActor":
        from miles.backends.fsdp_utils.actor import FSDPTrainRayActor

        value = FSDPTrainRayActor
    elif name == "load_fsdp_args":
        from miles.backends.fsdp_utils.arguments import load_fsdp_args

        value = load_fsdp_args
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


logging.getLogger().setLevel(logging.WARNING)
