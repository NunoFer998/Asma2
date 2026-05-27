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

from envs.finite_fuel_wrapper import FiniteFuelWrapper
from envs.custom_lunar_lander import CustomLunarLander
from evaluate import full_evaluation


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ppo_custom_lunar_lander"


# ═══════════════════════════════════════════════════════════════════════
#  Config tag — encodes hyperparameters into folder / file names
# ═══════════════════════════════════════════════════════════════════════

def make_config_tag(
    n_steps: int = 2048,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.01,
    lr_decay: bool = False,
) -> str:
    """Return a short, filesystem-safe string that uniquely identifies a run.

    Example: ``ns2048_bs64_lr3e-04_ec0.01``  or  ``ns1024_bs128_lr1e-03_ec0.05_decay``
    """
    lr_str = f"{learning_rate:.0e}".replace("+", "")  # "3e-04"
    tag = f"ns{n_steps}_bs{batch_size}_lr{lr_str}_ec{ent_coef}"
    if lr_decay:
        tag += "_decay"
    return tag


# ═══════════════════════════════════════════════════════════════════════
#  Environment helpers
# ═══════════════════════════════════════════════════════════════════════

def make_env(render_mode: str | None = None, seed: int | None = None) -> CustomLunarLander:
    # Se render_mode="human", o env base corre em rgb_array
    base_render_mode = "rgb_array" if render_mode == "human" else render_mode
    
    env = CustomLunarLander(render_mode=base_render_mode)
    env = FiniteFuelWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_vec_env(seed: int | None = None, n_envs: int = 4, render_mode: str | None = None) -> DummyVecEnv:
    """Create a vectorised training environment with *n_envs* sub-environments.

    Each sub-environment receives a different seed (seed, seed+1, …) so the
    agent sees diverse landing-pad positions during training.
    """
    def _make_factory(env_seed: int | None):
        def _factory() -> Monitor:
            env = make_env(render_mode=None, seed=env_seed)
            return Monitor(env)
        return _factory

    factories = [
        _make_factory(seed + i if seed is not None else None)
        for i in range(n_envs)
    ]
    return DummyVecEnv(factories)

# ═══════════════════════════════════════════════════════════════════════
#  Model building
# ═══════════════════════════════════════════════════════════════════════

def build_model(
    env,
    n_steps: int = 2048,
    batch_size: int = 64,
    learning_rate: float | Callable = 3e-4,
    ent_coef: float = 0.01,
    config_tag: str = "",
) -> PPO:
    """Build a PPO model with the given hyperparameters.

    Parameters
    ----------
    learning_rate : float or Callable
        Either a constant float or a schedule function (e.g. from
        ``get_linear_fn``).  When ``lr_decay=True`` in ``train_ppo_agent``,
        a linear decay schedule is passed here.
    """
    tb_dir = PROJECT_ROOT / "tensorboard"
    if config_tag:
        tb_dir = tb_dir / config_tag

    return PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=str(tb_dir),
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=ent_coef,
        clip_range=0.2,
        learning_rate=learning_rate,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════

def train_ppo_agent(
    total_timesteps: int = 500_000,
    model_path: Path | None = None,
    eval_frequency: int = 50_000,
    seed: int = 42,
    render_mode: str | None = None,
    continue_training: bool = False,
    # ── Tunable hyperparameters ──
    n_steps: int = 2048,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.01,
    lr_decay: bool = False,
) -> Path:
    config_tag = make_config_tag(n_steps, batch_size, learning_rate, ent_coef, lr_decay)

    # ── Build output paths organised by config ──────────────────
    if model_path is None or model_path == DEFAULT_MODEL_PATH:
        output_path = PROJECT_ROOT / "models" / config_tag / "model"
    else:
        output_path = model_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 60}")
    print(f"  Training config: {config_tag}")
    print(f"  Output dir     : {output_path.parent}")
    print(f"{'═' * 60}\n")

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
        tb_dir = PROJECT_ROOT / "tensorboard" / config_tag
        model.tensorboard_log = str(tb_dir)
    else:
        if continue_training:
            print(f"No existing model at {saved_file}, starting fresh.")

        # ── Resolve learning rate (constant or decay schedule) ──
        lr: float | Callable = learning_rate
        if lr_decay:
            from stable_baselines3.common.utils import get_linear_fn
            lr = get_linear_fn(learning_rate, 0.0, 1.0)
            print(f"  Using linear LR decay: {learning_rate} → 0")

        model = build_model(
            train_env,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=lr,
            ent_coef=ent_coef,
            config_tag=config_tag,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(eval_frequency, 1),
        save_path=str(output_path.parent / "checkpoints"),
        name_prefix=config_tag,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path.parent / "best_model"),
        log_path=str(output_path.parent / "eval_logs"),
        eval_freq=max(eval_frequency, 1),
        deterministic=True,
        render=(render_mode is not None),
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=not continue_training,
    )
    model.save(str(output_path))
    print(f"Model saved to {output_path.with_suffix('.zip')}")

    train_env.close()
    eval_env.close()
    return output_path.with_suffix(".zip")


def load_model(model_path: Path | None = None) -> PPO:
    path = (model_path or DEFAULT_MODEL_PATH).with_suffix(".zip")
    return PPO.load(str(path))


# ═══════════════════════════════════════════════════════════════════════
#  Policy execution & logging
# ═══════════════════════════════════════════════════════════════════════

def run_policy(
    model: PPO,
    episodes: int = 5,
    render_mode: str | None = "human",
    seed: int | None = None,
    save_log: bool = True,
    config_tag: str = "",
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
                "pad_chunk": getattr(env.unwrapped, "_pad_idx", None),
                "steps": steps,
            }
            all_episodes.append(episode_data)
            print(f"Episode {episode + 1}: total_reward={total_reward:.2f}, pad_chunk={getattr(env.unwrapped, '_pad_idx', 'n/a')}")
    finally:
        env.close()

    if save_log:
        _save_episode_log(all_episodes, config_tag=config_tag)

    return all_episodes


def _save_episode_log(episodes: list[dict], prefix: str = "episode_actions", config_tag: str = "") -> Path:
    """Persist episode data to a timestamped JSON file under ``logs/<config_tag>/``."""
    log_dir = PROJECT_ROOT / "logs"
    if config_tag:
        log_dir = log_dir / config_tag
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{prefix}_{timestamp}.json"
    log_file.write_text(json.dumps(episodes, indent=2), encoding="utf-8")
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
    summary_file.write_text(summary_text, encoding="utf-8")
    print(f"Summary saved to {summary_file}")
    print(summary_text)
    return summary_file


def run_random_baseline(
    episodes: int = 5,
    render_mode: str | None = "human",
    seed: int | None = None,
    save_log: bool = True,
    config_tag: str = "",
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
                "pad_chunk": getattr(env.unwrapped, "_pad_idx", None),
                "steps": steps,
            }
            all_episodes.append(episode_data)
            print(f"[Random] Episode {episode + 1}: total_reward={total_reward:.2f}, pad_chunk={getattr(env.unwrapped, '_pad_idx', 'n/a')}")
    finally:
        env.close()

    if save_log:
        _save_episode_log(all_episodes, prefix="random_baseline", config_tag=config_tag)

    return all_episodes


# ═══════════════════════════════════════════════════════════════════════
#  Train + Run + Evaluate pipeline
# ═══════════════════════════════════════════════════════════════════════

def train_then_run(
    total_timesteps: int = 500_000,
    episodes: int = 5,
    model_path: Path | None = None,
    seed: int = 42,
    render_mode: str | None = "human",
    continue_training: bool = False,
    eval_episodes: int = 50,
    # ── Tunable hyperparameters ──
    n_steps: int = 2048,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.01,
    lr_decay: bool = False,
) -> Path:
    config_tag = make_config_tag(n_steps, batch_size, learning_rate, ent_coef, lr_decay)

    saved_model_path = train_ppo_agent(
        total_timesteps=total_timesteps,
        model_path=model_path,
        seed=seed,
        continue_training=continue_training,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        lr_decay=lr_decay,
    )
    model = load_model(saved_model_path)
    run_policy(model, episodes=episodes, render_mode=render_mode, seed=seed, config_tag=config_tag)

    # ── Automatic post-training evaluation ──
    print("\n" + "=" * 55)
    print("  STARTING AUTOMATIC POST-TRAINING EVALUATION")
    print("=" * 55)
    full_evaluation(
        model=model,
        episodes=eval_episodes,
        seed=seed,
        model_dir=saved_model_path.parent,
        config_tag=config_tag,
    )

    return saved_model_path