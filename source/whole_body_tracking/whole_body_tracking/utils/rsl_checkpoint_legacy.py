"""Convert pre-rsl-rl-4 checkpoints (``model_state_dict``) for current runners."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rsl_rl.runners import OnPolicyRunner


def is_legacy_rsl_checkpoint(loaded: dict) -> bool:
    return "model_state_dict" in loaded and "actor_state_dict" not in loaded


def convert_legacy_rsl_checkpoint(loaded: dict) -> dict:
    """In-place upgrade of a legacy checkpoint dict for ``PPO.load`` (rsl-rl >= 4).

    Legacy checkpoints store a flat ``ActorCritic`` as ``model_state_dict`` with keys
    ``actor.*``, ``critic.*``, and ``std``. New checkpoints use separate
    ``actor_state_dict`` / ``critic_state_dict`` with ``mlp.*`` and ``distribution.*``.

    .. note::
        Observation normalizer buffers are not present in typical legacy saves; they stay
        at initialization unless you load a newer-format checkpoint.

    Returns:
        The same dict (mutated) for chaining.
    """
    if not is_legacy_rsl_checkpoint(loaded):
        return loaded

    ms: dict = loaded.pop("model_state_dict")
    actor_sd: dict = {}
    critic_sd: dict = {}
    for key, tensor in ms.items():
        if key == "std":
            actor_sd["distribution.std_param"] = tensor
        elif key.startswith("actor."):
            actor_sd["mlp." + key[len("actor.") :]] = tensor
        elif key.startswith("critic."):
            critic_sd["mlp." + key[len("critic.") :]] = tensor
        else:
            raise ValueError(f"Unexpected key in legacy model_state_dict: {key}")

    loaded["actor_state_dict"] = actor_sd
    loaded["critic_state_dict"] = critic_sd
    return loaded


def load_on_policy_runner_checkpoint(
    runner: "OnPolicyRunner", resume_path: str, device: str, *, weights_only: bool = False
) -> None:
    """Load a checkpoint file into an :class:`rsl_rl.runners.OnPolicyRunner` (supports legacy saves).

    Supports both rsl-rl v3 (``OnPolicyRunner.load``) and rsl-rl >= 4 (``PPO.load``).
    """
    # rsl-rl v3: PPO has no .load(); OnPolicyRunner.load(path) handles everything.
    if not hasattr(runner.alg, "load"):
        runner.load(resume_path, load_optimizer=False, map_location=device)
        print(f"[INFO]: Loaded checkpoint via OnPolicyRunner.load (rsl-rl v3): {resume_path}")
        return

    # rsl-rl >= 4 path
    loaded = torch.load(resume_path, map_location=device, weights_only=weights_only)
    if is_legacy_rsl_checkpoint(loaded):
        convert_legacy_rsl_checkpoint(loaded)
        print("[INFO]: Loaded legacy rsl-rl checkpoint (model_state_dict); converted for rsl-rl >= 4.")
        print(
            "[WARN]: Legacy save has no obs-normalizer state; if training used empirical observation normalization,"
            " behaviour may suffer. Prefer checkpoints saved with the same rsl-rl major version."
        )
        load_cfg = {"actor": True, "critic": True, "optimizer": False, "iteration": True, "rnd": True}
        load_iteration = runner.alg.load(loaded, load_cfg, strict=False)
    else:
        load_iteration = runner.alg.load(loaded, None, strict=True)
    if load_iteration and loaded.get("iter") is not None:
        runner.current_learning_iteration = loaded["iter"]
