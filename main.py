from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl_agent import DEFAULT_MODEL_PATH, load_model, run_policy, run_random_baseline, train_then_run
from evaluate import full_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or run a PPO agent on the custom Lunar Lander environment.")
    parser.add_argument("--train", action="store_true", help="Train a new PPO agent before running it.")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Number of training timesteps.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of demo episodes to render.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training and evaluation.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path used to save or load the PPO model.",
    )
    parser.add_argument("--no-display", action="store_true", help="Run without opening a display (headless).")
    parser.add_argument("--random", action="store_true", help="Run a random-action baseline instead of the trained agent.")
    parser.add_argument("--continue", dest="continue_training", action="store_true",
                        help="Continue training from an existing model instead of starting fresh (use with --train).")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run a comprehensive evaluation on an existing model (reports + plots).")
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Number of episodes for the comprehensive evaluation (default: 50).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.random:
        print("Running random baseline...")
        run_random_baseline(
            episodes=args.episodes,
            render_mode=None if args.no_display else "human",
            seed=args.seed,
        )
        return

    if args.evaluate:
        model_file = args.model_path.with_suffix(".zip")
        if not model_file.exists():
            print(f"Error: No model found at {model_file}. Train one first with --train.")
            sys.exit(1)
        print(f"Loading model from {model_file} for evaluation ...")
        model = load_model(args.model_path)
        full_evaluation(
            model=model,
            episodes=args.eval_episodes,
            seed=args.seed,
            model_dir=args.model_path.parent,
        )
        return

    if args.train:
        train_then_run(
            total_timesteps=args.timesteps,
            episodes=args.episodes,
            model_path=args.model_path,
            seed=args.seed,
            render_mode=None if args.no_display else "human",
            continue_training=args.continue_training,
            eval_episodes=args.eval_episodes,
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
            render_mode=None if args.no_display else "human",
            eval_episodes=args.eval_episodes,
        )
        return

    model = load_model(args.model_path)
    run_policy(model, episodes=args.episodes, render_mode=None if args.no_display else "human", seed=args.seed)


if __name__ == "__main__":
    main()