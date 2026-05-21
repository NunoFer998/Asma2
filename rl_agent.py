from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.custom_lunar_lander import CustomLunarLander


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ppo_custom_lunar_lander"


def make_env(render_mode: str | None = None, seed: int | None = None) -> CustomLunarLander:
    env = CustomLunarLander(render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_vec_env(seed: int | None = None, render_mode: str | None = None) -> DummyVecEnv:
    def _factory() -> Monitor:
        # training envs should not render for performance
        env = make_env(render_mode=None, seed=seed)
        return Monitor(env)

    return DummyVecEnv([_factory])


def build_model(env) -> PPO:
    return PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=str(PROJECT_ROOT / "tensorboard"),
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        learning_rate=3e-4,
    )


def train_ppo_agent(
    total_timesteps: int = 500_000,
    model_path: Path | None = None,
    eval_frequency: int = 50_000,
    seed: int = 42,
    render_mode: str | None = None,
    continue_training: bool = False,
) -> Path:
    output_path = model_path or DEFAULT_MODEL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    validation_env = make_env(render_mode=render_mode, seed=seed)
    check_env(validation_env, warn=True)
    validation_env.close()
    train_env = make_vec_env(seed=seed)
    # use a single env for evaluation so rendering opens a window
    eval_env = make_env(render_mode=render_mode, seed=seed + 1)

    saved_file = output_path.with_suffix(".zip")
    if continue_training and saved_file.exists():
        print(f"Continuing training from {saved_file}")
        model = PPO.load(str(output_path), env=train_env)
    else:
        if continue_training:
            print(f"No existing model at {saved_file}, starting fresh.")
        model = build_model(train_env)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(eval_frequency, 1),
        save_path=str(output_path.parent / "checkpoints"),
        name_prefix="ppo_custom_lunar_lander",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path.parent / "best_model"),
        log_path=str(output_path.parent / "eval_logs"),
        eval_freq=max(eval_frequency, 1),
        deterministic=True,
        render=(render_mode is not None),
    )

    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
    model.save(str(output_path))

    train_env.close()
    eval_env.close()
    return output_path.with_suffix(".zip")


def load_model(model_path: Path | None = None) -> PPO:
    path = (model_path or DEFAULT_MODEL_PATH).with_suffix(".zip")
    return PPO.load(str(path))


def run_policy(
    model: PPO,
    episodes: int = 5,
    render_mode: str | None = "human",
    seed: int | None = None,
    save_log: bool = True,
) -> list[dict]:
    """Run the policy for *episodes* episodes and optionally save a JSON log.

    Returns a list of episode dictionaries, each containing the actions,
    observations, rewards, and summary information.
    """
    env = make_env(render_mode=render_mode, seed=seed)
    all_episodes: list[dict] = []

    try:
        for episode in range(episodes):
            observation, info = env.reset(seed=None if seed is None else seed + episode)
            if render_mode is not None:
                try:
                    env.render()
                except Exception:
                    pass
            terminated = False
            truncated = False
            total_reward = 0.0
            step_count = 0
            steps: list[dict] = []

            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                if render_mode is not None:
                    try:
                        env.render()
                    except Exception:
                        pass
                total_reward += float(reward)
                step_count += 1

                # Record step data
                steps.append({
                    "step": step_count,
                    "action": int(action) if hasattr(action, "item") else action,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "observation": observation.tolist(),
                })

            episode_data = {
                "episode": episode + 1,
                "total_reward": round(total_reward, 4),
                "total_steps": step_count,
                "pad_chunk": getattr(env, "_pad_idx", None),
                "steps": steps,
            }
            all_episodes.append(episode_data)
            print(f"Episode {episode + 1}: total_reward={total_reward:.2f}, pad_chunk={getattr(env, '_pad_idx', 'n/a')}")
    finally:
        env.close()

    if save_log:
        _save_episode_log(all_episodes)

    return all_episodes


def _save_episode_log(episodes: list[dict], prefix: str = "episode_actions") -> Path:
    """Persist episode data to a timestamped JSON file under ``logs/``."""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{prefix}_{timestamp}.json"
    log_file.write_text(json.dumps(episodes, indent=2))
    print(f"\nEpisode actions saved to {log_file}")

    # Also save a human-readable summary
    _save_summary(episodes, log_dir, prefix, timestamp)

    return log_file


def _save_summary(episodes: list[dict], log_dir: Path, prefix: str, timestamp: str) -> Path:
    """Write a concise, human-readable summary of the run to a ``.txt`` file."""
    import statistics

    rewards = [ep["total_reward"] for ep in episodes]
    steps   = [ep["total_steps"]  for ep in episodes]

    lines: list[str] = []
    lines.append(f"{'=' * 50}")
    lines.append(f"  Run Summary  —  {prefix}")
    lines.append(f"  {timestamp[:8]}  {timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}")
    lines.append(f"{'=' * 50}")
    lines.append("")

    # ── Per-episode table ──
    lines.append(f"  {'Episode':<10} {'Reward':>10} {'Steps':>8} {'Pad chunk':>10}")
    lines.append(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
    for ep in episodes:
        pad = ep.get("pad_chunk", "n/a")
        lines.append(f"  {ep['episode']:<10} {ep['total_reward']:>10.2f} {ep['total_steps']:>8} {str(pad):>10}")

    lines.append("")
    lines.append(f"{'─' * 50}")
    lines.append(f"  AVERAGES")
    lines.append(f"{'─' * 50}")
    lines.append(f"  Mean reward    : {statistics.mean(rewards):>10.2f}")
    lines.append(f"  Std  reward    : {statistics.stdev(rewards):>10.2f}" if len(rewards) > 1 else "  Std  reward    :        n/a")
    lines.append(f"  Min  reward    : {min(rewards):>10.2f}")
    lines.append(f"  Max  reward    : {max(rewards):>10.2f}")
    lines.append("")
    lines.append(f"  Mean steps     : {statistics.mean(steps):>10.1f}")
    lines.append(f"  Min  steps     : {min(steps):>10}")
    lines.append(f"  Max  steps     : {max(steps):>10}")
    lines.append(f"  Total episodes : {len(episodes):>10}")
    lines.append(f"{'=' * 50}")

    summary_text = "\n".join(lines) + "\n"
    summary_file = log_dir / f"{prefix}_summary_{timestamp}.txt"
    summary_file.write_text(summary_text)
    print(f"Summary saved to {summary_file}")
    print(summary_text)
    return summary_file


def run_random_baseline(
    episodes: int = 5,
    render_mode: str | None = "human",
    seed: int | None = None,
    save_log: bool = True,
) -> list[dict]:
    """Run episodes choosing actions uniformly at random (baseline).

    Uses the same logging format as ``run_policy`` so results are directly
    comparable.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    env = make_env(render_mode=render_mode, seed=seed)
    all_episodes: list[dict] = []

    try:
        for episode in range(episodes):
            observation, info = env.reset(seed=None if seed is None else seed + episode)
            if render_mode is not None:
                try:
                    env.render()
                except Exception:
                    pass
            terminated = False
            truncated = False
            total_reward = 0.0
            step_count = 0
            steps: list[dict] = []

            while not (terminated or truncated):
                action = int(rng.integers(env.action_space.n))
                observation, reward, terminated, truncated, info = env.step(action)
                if render_mode is not None:
                    try:
                        env.render()
                    except Exception:
                        pass
                total_reward += float(reward)
                step_count += 1

                steps.append({
                    "step": step_count,
                    "action": action,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "observation": observation.tolist(),
                })

            episode_data = {
                "episode": episode + 1,
                "total_reward": round(total_reward, 4),
                "total_steps": step_count,
                "pad_chunk": getattr(env, "_pad_idx", None),
                "steps": steps,
            }
            all_episodes.append(episode_data)
            print(f"[Random] Episode {episode + 1}: total_reward={total_reward:.2f}, pad_chunk={getattr(env, '_pad_idx', 'n/a')}")
    finally:
        env.close()

    if save_log:
        _save_episode_log(all_episodes, prefix="random_baseline")

    return all_episodes


def train_then_run(
    total_timesteps: int = 500_000,
    episodes: int = 5,
    model_path: Path | None = None,
    seed: int = 42,
    render_mode: str | None = "human",
    continue_training: bool = False,
) -> Path:
    saved_model_path = train_ppo_agent(
        total_timesteps=total_timesteps,
        model_path=model_path,
        seed=seed,
        continue_training=continue_training,
    )
    model = load_model(saved_model_path)
    run_policy(model, episodes=episodes, render_mode=render_mode, seed=seed)
    return saved_model_path