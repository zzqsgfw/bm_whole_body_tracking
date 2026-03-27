"""Roll out an RSL-RL checkpoint and export G1 full-body trajectories as .pt.

The exported schema matches the g1_fullbody motion-generation dataset:
    - command
    - motion_anchor_pos_b
    - motion_anchor_ori_b
    - base_lin_vel
    - base_ang_vel
    - joint_pos
    - joint_vel
    - last_action
    - expert_action

Each saved file contains a list[dict], where each dict is one trajectory.
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Export rollout trajectories from an RSL-RL checkpoint.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Task name.")
parser.add_argument("--motion_file", type=str, default=None, help="Override motion file.")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Absolute or relative path to checkpoint.")
parser.add_argument("--save_dir", type=str, default=None, help="Directory to save exported .pt trajectories.")
parser.add_argument("--save_name", type=str, default=None, help="Output .pt filename.")
parser.add_argument("--num_trajectories", type=int, default=1, help="Number of trajectories to export.")
parser.add_argument("--max_steps", type=int, default=None, help="Optional cap on rollout steps.")
parser.add_argument("--save_partial", action="store_true", default=False, help="Also save unfinished trajectories.")
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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAVE_DIR = REPO_ROOT / "logs" / "trajs"


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


def _resolve_resume_path(agent_cfg: RslRlOnPolicyRunnerCfg) -> tuple[str, Path]:
    if args_cli.checkpoint_path is not None:
        resume_path = Path(args_cli.checkpoint_path).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        return str(resume_path), resume_path.parent

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    return resume_path, Path(os.path.dirname(resume_path))


def _collect_clean_frame(raw_env, expert_action: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "command": isaac_mdp.generated_commands(raw_env, "motion").detach().cpu(),
        "motion_anchor_pos_b": tracking_mdp.motion_anchor_pos_b(raw_env, "motion").detach().cpu(),
        "motion_anchor_ori_b": tracking_mdp.motion_anchor_ori_b(raw_env, "motion").detach().cpu(),
        "base_lin_vel": isaac_mdp.base_lin_vel(raw_env).detach().cpu(),
        "base_ang_vel": isaac_mdp.base_ang_vel(raw_env).detach().cpu(),
        "joint_pos": isaac_mdp.joint_pos_rel(raw_env).detach().cpu(),
        "joint_vel": isaac_mdp.joint_vel_rel(raw_env).detach().cpu(),
        "last_action": isaac_mdp.last_action(raw_env).detach().cpu(),
        "expert_action": expert_action.detach().cpu(),
    }


def _new_buffer() -> dict[str, list[torch.Tensor]]:
    return {
        "command": [],
        "motion_anchor_pos_b": [],
        "motion_anchor_ori_b": [],
        "base_lin_vel": [],
        "base_ang_vel": [],
        "joint_pos": [],
        "joint_vel": [],
        "last_action": [],
        "expert_action": [],
    }


def _finalize_trajectory(
    buffers: list[dict[str, list[torch.Tensor]]],
    env_id: int,
    save_name: str,
    traj_index: int,
    checkpoint_path: str,
    motion_file: str | None,
    terminated: bool,
    wrapped: bool,
):
    if len(buffers[env_id]["expert_action"]) == 0:
        return None

    traj = {key: torch.stack(value, dim=0) for key, value in buffers[env_id].items()}
    traj["metadata"] = {
        "total_frames": traj["expert_action"].shape[0],
        "recording_name": save_name,
        "trajectory_id": f"{save_name}_traj{traj_index:04d}_env{env_id}",
        "schema": "g1_fullbody_v1",
        "condition_dim": 160,
        "action_dim": 29,
        "env_id": env_id,
        "source_checkpoint": checkpoint_path,
        "motion_file": motion_file,
        "terminated": terminated,
        "wrapped_motion": wrapped,
    }
    buffers[env_id] = _new_buffer()
    return traj


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs

    resume_path, run_dir = _resolve_resume_path(agent_cfg)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    if args_cli.motion_file is not None:
        env_cfg.commands.motion.motion_file = args_cli.motion_file
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
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    raw_env = env.unwrapped
    motion_cmd = raw_env.command_manager.get_term("motion")
    default_max_steps = int(motion_cmd.motion.time_step_total)
    max_steps = args_cli.max_steps if args_cli.max_steps is not None else default_max_steps

    save_dir = Path(args_cli.save_dir).expanduser() if args_cli.save_dir else DEFAULT_SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_stem = Path(resume_path).stem
    run_name = run_dir.name
    save_name = args_cli.save_name or f"{run_name}__{checkpoint_stem}.pt"
    if not save_name.endswith(".pt"):
        save_name += ".pt"
    output_path = save_dir / save_name

    print_dict(
        {
            "num_envs": args_cli.num_envs,
            "num_trajectories": args_cli.num_trajectories,
            "max_steps": max_steps,
            "output_path": str(output_path),
            "motion_file": env_cfg.commands.motion.motion_file,
        },
        nesting=4,
    )

    obs, _ = env.get_observations()
    buffers = [_new_buffer() for _ in range(args_cli.num_envs)]
    trajectories = []
    steps = 0

    while simulation_app.is_running() and steps < max_steps and len(trajectories) < args_cli.num_trajectories:
        with torch.inference_mode():
            prev_time_steps = motion_cmd.time_steps.detach().clone()
            actions = policy(obs)
            frame = _collect_clean_frame(raw_env, actions)

            for env_id in range(args_cli.num_envs):
                for key, value in frame.items():
                    buffers[env_id][key].append(value[env_id])

            obs, _, dones, _ = env.step(actions)
            next_time_steps = motion_cmd.time_steps.detach().clone()
            wrapped = next_time_steps <= prev_time_steps
            done_mask = dones.to(dtype=torch.bool).detach().cpu()
            wrapped_mask = wrapped.detach().cpu()

            for env_id in range(args_cli.num_envs):
                if bool(done_mask[env_id]) or bool(wrapped_mask[env_id]):
                    traj = _finalize_trajectory(
                        buffers,
                        env_id,
                        output_path.stem,
                        len(trajectories),
                        resume_path,
                        env_cfg.commands.motion.motion_file,
                        terminated=bool(done_mask[env_id]),
                        wrapped=bool(wrapped_mask[env_id]),
                    )
                    if traj is not None:
                        trajectories.append(traj)
                        if len(trajectories) >= args_cli.num_trajectories:
                            break
        steps += 1

    if args_cli.save_partial and len(trajectories) < args_cli.num_trajectories:
        for env_id in range(args_cli.num_envs):
            traj = _finalize_trajectory(
                buffers,
                env_id,
                output_path.stem,
                len(trajectories),
                resume_path,
                env_cfg.commands.motion.motion_file,
                terminated=False,
                wrapped=False,
            )
            if traj is not None:
                trajectories.append(traj)
            if len(trajectories) >= args_cli.num_trajectories:
                break

    torch.save(trajectories, output_path)
    print(f"[INFO] Saved {len(trajectories)} trajectories to: {output_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
