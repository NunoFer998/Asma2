from __future__ import annotations

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
) -> Path:
    output_path = model_path or DEFAULT_MODEL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    validation_env = make_env(render_mode=render_mode, seed=seed)
    check_env(validation_env, warn=True)
    validation_env.close()
    train_env = make_vec_env(seed=seed)
    # use a single env for evaluation so rendering opens a window
    eval_env = make_env(render_mode=render_mode, seed=seed + 1)
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
) -> None:
    env = make_env(render_mode=render_mode, seed=seed)
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

            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                if render_mode is not None:
                    try:
                        env.render()
                    except Exception:
                        pass
                total_reward += float(reward)

            print(f"Episode {episode + 1}: total_reward={total_reward:.2f}, pad_chunk={getattr(env, '_pad_idx', 'n/a')}")
    finally:
        env.close()


def train_then_run(
    total_timesteps: int = 500_000,
    episodes: int = 5,
    model_path: Path | None = None,
    seed: int = 42,
    render_mode: str | None = "human",
) -> Path:
    saved_model_path = train_ppo_agent(
        total_timesteps=total_timesteps,
        model_path=model_path,
        seed=seed,
    )
    model = load_model(saved_model_path)
    run_policy(model, episodes=episodes, render_mode=render_mode, seed=seed)
    return saved_model_path