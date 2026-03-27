import os
from pathlib import Path

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def _is_git_tracked_path(path: str) -> bool:
    """Return whether the given file path lives inside a git repository checkout."""
    current = Path(path).resolve().parent
    for parent in (current, *current.parents):
        git_path = parent / ".git"
        if git_path.exists():
            return True
    return False


class MyOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        self.logger.git_status_repos = [
            path for path in self.logger.git_status_repos if _is_git_tracked_path(path)
        ]

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if getattr(self.logger, "logger_type", None) == "wandb":
            import wandb

            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            self.export_policy_to_onnx(policy_path, filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.logger.git_status_repos = [
            path for path in self.logger.git_status_repos if _is_git_tracked_path(path)
        ]
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if getattr(self.logger, "logger_type", None) == "wandb":
            import wandb

            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped,
                self.alg.get_policy(),
                path=policy_path,
                filename=filename,
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
