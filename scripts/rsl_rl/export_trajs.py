"""DART-style rollout: inject per-env action noise for BC generalisation.

Env 0 is ALWAYS noise-free (the "clean" reference trajectory).
Envs 1..N-1 each get i.i.d. Gaussian noise on top of the policy output.
The *noisy* action is what gets executed AND recorded as ``expert_action``,
while the raw policy output is saved separately as ``policy_action`` so the
student can optionally regress to either target.

Trajectory list layout (deterministic):
    trajectories[0]           -> env-0, noise_std = 0  (clean)
    trajectories[1..N-1]      -> env-1..N-1, noise_std > 0

Schema per trajectory dict (g1_fullbody_v3_dart):
    command              (T, 15)
    base_pos_w           (T, 3)    # world-frame root position (for MuJoCo viewer)
    base_quat_w          (T, 4)    # world-frame root quaternion [w,x,y,z]
    motion_anchor_pos_b  (T, 3)    # motion anchor relative to robot anchor (body frame)
    motion_anchor_ori_b  (T, 6)    # motion anchor relative orientation (Rot6D)
    body_pos_b           (T, 42)   # 14 bodies * 3 (Cartesian pos in anchor frame)
    body_ori_b           (T, 84)   # 14 bodies * 6 (Rot6D in anchor frame)
    base_lin_vel         (T, 3)
    base_ang_vel         (T, 3)
    joint_pos            (T, 29)   # joint_pos_rel (relative to default)
    joint_vel            (T, 29)
    last_action          (T, 29)
    policy_action        (T, 29)   # raw policy output (no noise)
    expert_action        (T, 29)   # = policy_action + noise  (executed)
    metadata             dict
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="DART-style trajectory export from an RSL-RL checkpoint.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments (env-0 is noise-free).")
parser.add_argument("--task", type=str, default=None, help="Task name.")
parser.add_argument("--motion_file", type=str, default=None, help="Override motion file.")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Absolute or relative path to checkpoint.")
parser.add_argument("--checkpoint", type=str, default=None, help="Alias of --checkpoint_path for consistency with play.py.")
parser.add_argument("--save_dir", type=str, default=None, help="Directory to save exported .pt trajectories.")
parser.add_argument("--save_name", type=str, default=None, help="Output .pt filename.")
parser.add_argument("--max_steps", type=int, default=None, help="Optional cap on rollout steps (default: full motion length).")
parser.add_argument("--noise_std", type=float, default=0.05, help="Action-space Gaussian noise std for envs 1..N-1.")
parser.add_argument("--action_clip", type=float, default=1.0, help="Clip noisy actions to [-clip, clip].")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import yaml

import isaaclab.envs.mdp as isaac_mdp
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401
import whole_body_tracking.tasks.tracking.mdp as tracking_mdp
from whole_body_tracking.utils.rsl_checkpoint_legacy import load_on_policy_runner_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAVE_DIR = REPO_ROOT / "logs" / "trajs"

FRAME_KEYS = [
    "command",
    "base_pos_w",
    "base_quat_w",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "body_pos_b",
    "body_ori_b",
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "last_action",
    "policy_action",
    "expert_action",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_motion_file_from_run_dir(run_dir: Path) -> str | None:
    env_yaml = run_dir / "params" / "env.yaml"
    if not env_yaml.is_file():
        return None
    with env_yaml.open("r", encoding="utf-8") as f:
        data = yaml.unsafe_load(f)
    motion_file = (
        data.get("commands", {})
        .get("motion", {})
        .get("motion_file")
    )
    if motion_file is None:
        return None
    motion_path = Path(motion_file)
    if motion_path.is_absolute():
        return str(motion_path)
    return str((REPO_ROOT / motion_path).resolve())


def _resolve_motion_file_path(motion_file: str) -> str:
    """Resolve motion .npz path similarly to play.py."""
    raw = motion_file.strip()
    expanded = os.path.expanduser(raw)
    for candidate in (expanded, os.path.abspath(expanded)):
        if os.path.isfile(candidate):
            return candidate
    rel = raw.replace("\\", "/").lstrip("/")
    for prefix in ("whole_body_tracking/", "./whole_body_tracking/"):
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
            break
    cand = REPO_ROOT / rel
    if cand.is_file():
        return str(cand.resolve())
    return raw


def _resolve_resume_path(agent_cfg: RslRlOnPolicyRunnerCfg) -> tuple[str, Path]:
    checkpoint_arg = args_cli.checkpoint_path or args_cli.checkpoint
    if checkpoint_arg is not None:
        resume_path = Path(checkpoint_arg).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        return str(resume_path), resume_path.parent
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    return resume_path, Path(os.path.dirname(resume_path))


def _build_noise_mask(num_envs: int, noise_std: float, device: torch.device) -> torch.Tensor:
    """Per-env noise scale: env-0 = 0, envs 1..N-1 = noise_std."""
    mask = torch.full((num_envs, 1), noise_std, device=device)
    mask[0] = 0.0
    return mask


def _inject_noise(
    policy_action: torch.Tensor,
    noise_mask: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    noise = torch.randn_like(policy_action) * noise_mask
    return (policy_action + noise).clamp(-clip, clip)


def _probe_frame_dims(raw_env) -> dict[str, int]:
    """Probe observation dims once."""
    with torch.inference_mode():
        return {
            "command": isaac_mdp.generated_commands(raw_env, "motion").shape[-1],
            "base_pos_w": isaac_mdp.root_pos_w(raw_env).shape[-1],
            "base_quat_w": isaac_mdp.root_quat_w(raw_env).shape[-1],
            "motion_anchor_pos_b": tracking_mdp.motion_anchor_pos_b(raw_env, "motion").shape[-1],
            "motion_anchor_ori_b": tracking_mdp.motion_anchor_ori_b(raw_env, "motion").shape[-1],
            "body_pos_b": tracking_mdp.robot_body_pos_b(raw_env, "motion").shape[-1],
            "body_ori_b": tracking_mdp.robot_body_ori_b(raw_env, "motion").shape[-1],
            "base_lin_vel": isaac_mdp.base_lin_vel(raw_env).shape[-1],
            "base_ang_vel": isaac_mdp.base_ang_vel(raw_env).shape[-1],
            "joint_pos": isaac_mdp.joint_pos_rel(raw_env).shape[-1],
            "joint_vel": isaac_mdp.joint_vel_rel(raw_env).shape[-1],
            "last_action": isaac_mdp.last_action(raw_env).shape[-1],
        }


def _alloc_chunk(chunk_size: int, num_envs: int, frame_dims: dict[str, int], action_dim: int,
                 device: torch.device) -> dict[str, torch.Tensor]:
    """Allocate a GPU chunk buffer (chunk_size, num_envs, dim)."""
    buf = {}
    for key, dim in frame_dims.items():
        buf[key] = torch.empty(chunk_size, num_envs, dim, device=device)
    buf["policy_action"] = torch.empty(chunk_size, num_envs, action_dim, device=device)
    buf["expert_action"] = torch.empty(chunk_size, num_envs, action_dim, device=device)
    return buf


def _write_frame(buf: dict[str, torch.Tensor], idx: int, raw_env,
                 policy_action: torch.Tensor, expert_action: torch.Tensor):
    """Write one step into GPU chunk at position idx."""
    buf["command"][idx] = isaac_mdp.generated_commands(raw_env, "motion")
    buf["base_pos_w"][idx] = isaac_mdp.root_pos_w(raw_env)
    buf["base_quat_w"][idx] = isaac_mdp.root_quat_w(raw_env)
    buf["motion_anchor_pos_b"][idx] = tracking_mdp.motion_anchor_pos_b(raw_env, "motion")
    buf["motion_anchor_ori_b"][idx] = tracking_mdp.motion_anchor_ori_b(raw_env, "motion")
    buf["body_pos_b"][idx] = tracking_mdp.robot_body_pos_b(raw_env, "motion")
    buf["body_ori_b"][idx] = tracking_mdp.robot_body_ori_b(raw_env, "motion")
    buf["base_lin_vel"][idx] = isaac_mdp.base_lin_vel(raw_env)
    buf["base_ang_vel"][idx] = isaac_mdp.base_ang_vel(raw_env)
    buf["joint_pos"][idx] = isaac_mdp.joint_pos_rel(raw_env)
    buf["joint_vel"][idx] = isaac_mdp.joint_vel_rel(raw_env)
    buf["last_action"][idx] = isaac_mdp.last_action(raw_env)
    buf["policy_action"][idx] = policy_action
    buf["expert_action"][idx] = expert_action


def _flush_chunk(gpu_chunk: dict[str, torch.Tensor], used: int,
                 cpu_store: dict[str, list[torch.Tensor]]):
    """Async-copy filled portion of GPU chunk to CPU pinned memory."""
    for key, tensor in gpu_chunk.items():
        cpu_store[key].append(tensor[:used].to("cpu", non_blocking=True))
    torch.cuda.synchronize()


def _vram_usage_ratio(device: torch.device) -> tuple[float, int, int]:
    """Return (used/total ratio, used_mb, total_mb) for the given CUDA device."""
    idx = device.index if device.index is not None else 0
    used = torch.cuda.memory_allocated(idx)
    total = torch.cuda.get_device_properties(idx).total_memory
    return used / total, used >> 20, total >> 20


def _vram_is_dangerous(device: torch.device, threshold: float = 0.85) -> bool:
    """True if VRAM usage exceeds threshold (default 85%)."""
    ratio, _, _ = _vram_usage_ratio(device)
    return ratio > threshold


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs

    # Keep original episode_length (10s) — policy was trained with this horizon.
    # Disable random pushes during export for cleaner trajectories.
    env_cfg.events.push_robot = None

    resume_path, run_dir = _resolve_resume_path(agent_cfg)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    if args_cli.motion_file is not None:
        resolved_motion = _resolve_motion_file_path(args_cli.motion_file)
        print(f"[INFO] Using motion file from CLI: {resolved_motion}")
        env_cfg.commands.motion.motion_file = resolved_motion
    else:
        inferred_motion = _load_motion_file_from_run_dir(run_dir)
        if inferred_motion is not None:
            env_cfg.commands.motion.motion_file = inferred_motion
            print(f"[INFO] Inferred motion file: {inferred_motion}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    load_on_policy_runner_checkpoint(ppo_runner, resume_path, agent_cfg.device)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    raw_env = env.unwrapped
    motion_cmd = raw_env.command_manager.get_term("motion")
    default_max_steps = int(motion_cmd.motion.time_step_total)
    max_steps = args_cli.max_steps if args_cli.max_steps is not None else default_max_steps

    noise_std = args_cli.noise_std
    action_clip = args_cli.action_clip

    save_dir = Path(args_cli.save_dir).expanduser() if args_cli.save_dir else DEFAULT_SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_stem = Path(resume_path).stem
    run_name = run_dir.name
    save_name = args_cli.save_name or f"{run_name}__{checkpoint_stem}__dart.pt"
    if not save_name.endswith(".pt"):
        save_name += ".pt"
    output_path = save_dir / save_name

    task_name = args_cli.task
    noise_mask = _build_noise_mask(args_cli.num_envs, noise_std, env.unwrapped.device)

    print_dict(
        {
            "num_envs": args_cli.num_envs,
            "max_steps": max_steps,
            "noise_std": noise_std,
            "action_clip": action_clip,
            "task": task_name,
            "output_path": str(output_path),
            "motion_file": env_cfg.commands.motion.motion_file,
            "mode": "full_trajectory (no early termination)",
        },
        nesting=4,
    )

    # -----------------------------------------------------------------------
    # Chunked GPU rollout — all terminations disabled, all envs run max_steps.
    # GPU chunk buffer flushed to CPU periodically for VRAM safety.
    # -----------------------------------------------------------------------
    action_dim = 29
    num_envs = args_cli.num_envs
    CHUNK_STEPS = 512
    frame_dims = _probe_frame_dims(raw_env)
    device = env.unwrapped.device

    ratio_init, used_init, total_init = _vram_usage_ratio(device)
    print(f"[INFO] VRAM baseline: {used_init} MB / {total_init} MB ({ratio_init:.1%})")

    gpu_chunk = _alloc_chunk(CHUNK_STEPS, num_envs, frame_dims, action_dim, device)
    cpu_store: dict[str, list[torch.Tensor]] = {k: [] for k in FRAME_KEYS}

    chunk_mem_mb = sum(t.nelement() * t.element_size() for t in gpu_chunk.values()) / (1024 * 1024)
    print(f"[INFO] GPU chunk: {chunk_mem_mb:.1f} MB ({CHUNK_STEPS} steps x {num_envs} envs)")
    print(f"[INFO] Terminations DISABLED, episode_length=1e6, push_robot OFF")
    print(f"[INFO] Rollout: {num_envs} envs x {max_steps} steps")

    obs = env.get_observations()
    steps = 0
    chunk_idx = 0

    while simulation_app.is_running() and steps < max_steps:
        with torch.inference_mode():
            policy_action = policy(obs)
            expert_action = _inject_noise(policy_action, noise_mask, action_clip)
            _write_frame(gpu_chunk, chunk_idx, raw_env, policy_action, expert_action)
            obs, _, _, _ = env.step(expert_action)

        steps += 1
        chunk_idx += 1

        if chunk_idx >= CHUNK_STEPS:
            _flush_chunk(gpu_chunk, chunk_idx, cpu_store)
            chunk_idx = 0
            if steps % 1000 < CHUNK_STEPS:
                ratio, used, _ = _vram_usage_ratio(device)
                print(f"  step {steps}/{max_steps} (VRAM {used}MB {ratio:.1%})")

    if chunk_idx > 0:
        _flush_chunk(gpu_chunk, chunk_idx, cpu_store)

    del gpu_chunk
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Bulk GPU->CPU concat, then slice per-env
    # -----------------------------------------------------------------------
    print(f"\n[INFO] Rollout done ({steps} steps). Concatenating CPU buffers...")
    cpu_buf = {key: torch.cat(chunks, dim=0) for key, chunks in cpu_store.items()}
    del cpu_store

    print(f"[INFO] Building {num_envs} trajectory dicts...")
    trajectories = []
    # env-0 (clean) first
    for env_id in range(num_envs):
        traj = {key: cpu_buf[key][:, env_id, :] for key in FRAME_KEYS}
        env_noise = 0.0 if env_id == 0 else noise_std
        traj["metadata"] = {
            "total_frames": steps,
            "recording_name": output_path.stem,
            "trajectory_id": f"{output_path.stem}_traj{env_id:04d}_env{env_id}",
            "schema": "g1_fullbody_v3_dart",
            "condition_dim": 286,
            "action_dim": action_dim,
            "env_id": env_id,
            "noise_std": env_noise,
            "is_clean": env_id == 0,
            "task": task_name,
            "source_checkpoint": resume_path,
            "source_run_dir": run_name,
            "motion_file": env_cfg.commands.motion.motion_file,
        }
        trajectories.append(traj)
    del cpu_buf

    print(f"\n[INFO] Writing to disk: {output_path}")
    torch.save(trajectories, output_path)
    total_frames = steps * num_envs
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[INFO] Saved {len(trajectories)} trajectories ({total_frames} total frames, {file_size_mb:.1f} MB)")
    print(f"  trajectories[0] is_clean=True, {steps} frames")
    print(f"  task={task_name}  checkpoint={checkpoint_stem}  run={run_name}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
