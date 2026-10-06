"""FSDP2 backend, with lazy actor imports for model-only and argument consumers."""

__all__ = ["load_fsdp_args", "FSDPTrainRayActor"]


def __getattr__(name):
    # The Ray actor needs the distributed runtime; tensor/model utilities do not.
    if name == "FSDPTrainRayActor":
        from miles.backends.fsdp_utils.actor import FSDPTrainRayActor

        return FSDPTrainRayActor
    if name == "load_fsdp_args":
        from miles.backends.fsdp_utils.arguments import load_fsdp_args

        return load_fsdp_args
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
