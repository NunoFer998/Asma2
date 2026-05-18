from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl_agent import DEFAULT_MODEL_PATH, load_model, run_policy, train_then_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or run a PPO agent on the custom Lunar Lander environment.")
    parser.add_argument("--train", action="store_true", help="Train a new PPO agent before running it.")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Number of training timesteps.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes to run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training and evaluation.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path used to save or load the PPO model.",
    )
    parser.add_argument("--no-display", action="store_true", help="Run without opening a display (headless).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.train:
        train_then_run(
            total_timesteps=args.timesteps,
            episodes=args.episodes,
            model_path=args.model_path,
            seed=args.seed,
            render_mode=None if args.no_display else "human",
        )
        return

    model_file = args.model_path.with_suffix(".zip")
    if not model_file.exists():
        print(f"Model not found at {model_file}. Training a new PPO agent first.")
        train_then_run(
            total_timesteps=args.timesteps,
            episodes=args.episodes,
            model_path=args.model_path,
            seed=args.seed,
        )
        return

    model = load_model(args.model_path)
    run_policy(model, episodes=args.episodes, render_mode=None if args.no_display else "human", seed=args.seed)


if __name__ == "__main__":
    main()