"""Generation backends (diffusers, Automatic1111, ComfyUI) and the manifest builder."""

from .manifest import (  # noqa: F401
    build_all,
    build_ablation,
    build_calibration,
    build_generalization,
    build_main,
    build_migration,
    build_nonlinear,
    build_two_word,
    check_counts,
    calibrate_factor,
)
