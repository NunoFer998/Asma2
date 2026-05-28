from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rl_agent import DEFAULT_MODEL_PATH, load_model, run_policy, run_random_baseline, train_then_run, make_config_tag
from evaluate import full_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or run a PPO agent on the custom Lunar Lander environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default hyperparams
  python main.py --train --no-display

  # Train with custom hyperparams
  python main.py --train --no-display --n-steps 1024 --batch-size 128 --lr 1e-3 --ent-coef 0.05

  # Train with linear LR decay
  python main.py --train --no-display --lr 3e-4 --lr-decay

  # Train and compare against a previously trained model instead of random baseline
  python main.py --train --no-display --baseline ns2048_bs64_lr3e-04_ec0.01

  # Evaluate a specific model
  python main.py --evaluate --model-path models/ns1024_bs128_lr1e-03_ec0.05/model

  # Evaluate and compare against another model
  python main.py --evaluate --baseline ns2048_bs64_lr3e-04_ec0.01
""",
    )

    # ── Mode selection ──
    parser.add_argument("--train", action="store_true", help="Train a new PPO agent before running it.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run a comprehensive evaluation on an existing model (reports + plots).")
    parser.add_argument("--random", action="store_true", help="Run a random-action baseline instead of the trained agent.")
    parser.add_argument("--original-env", action="store_true",
                        help="Run against the original Gymnasium LunarLander environment instead of the custom wrapper.")

    # ── General options ──
    parser.add_argument("--timesteps", type=int, default=500_000, help="Number of training timesteps.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of demo episodes to render.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training and evaluation.")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Explicit path to save/load the PPO model (overrides auto-naming).")
    parser.add_argument("--no-display", action="store_true", help="Run without opening a display (headless).")
    parser.add_argument("--continue", dest="continue_training", action="store_true",
                        help="Continue training from an existing model instead of starting fresh (use with --train).")
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Number of episodes for the comprehensive evaluation (default: 50).")
    parser.add_argument("--baseline", type=str, default="",
                        help="Config tag of a previously trained model to use as the comparison baseline "
                             "instead of the random baseline (e.g. ns2048_bs64_lr3e-04_ec0.01). "
                             "Hyphens are converted to underscores automatically.")

    # ── Tunable hyperparameters ──
    hp = parser.add_argument_group("hyperparameters", "PPO hyperparameters for benchmarking")
    hp.add_argument("--n-steps", type=int, default=2048,
                    help="Rollout length per env per update (try: 1024, 2048, 4096).")
    hp.add_argument("--batch-size", type=int, default=64,
                    help="Minibatch size for SGD updates (try: 64, 128, 256).")
    hp.add_argument("--lr", type=float, default=3e-4,
                    help="Learning rate for Adam optimiser (try: 1e-4, 3e-4, 1e-3).")
    hp.add_argument("--ent-coef", type=float, default=0.01,
                    help="Entropy coefficient — controls exploration (try: 0.0, 0.01, 0.05).")
    hp.add_argument("--lr-decay", action="store_true",
                    help="Use linear LR decay from --lr to 0 over training.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Normalise baseline tag: allow hyphens on the CLI, convert to underscores
    baseline_tag = args.baseline.replace("-", "_") if args.baseline else ""

    # Build the config tag from the hyperparameters (used for file/folder naming)
    config_tag = make_config_tag(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        lr_decay=args.lr_decay,
        use_original_env=args.original_env,
    )

    # Resolve model path: explicit --model-path wins, otherwise auto-generate from config
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = DEFAULT_MODEL_PATH  # train_ppo_agent will auto-override with config_tag

    if args.random:
        print("Running random baseline...")
        run_random_baseline(
            episodes=args.episodes,
            render_mode=None if args.no_display else "human",
            seed=args.seed,
            config_tag=config_tag,
            use_original_env=args.original_env,
        )
        return

    if args.evaluate:
        # For evaluate, resolve the model path from config tag if not explicit
        if args.model_path is None:
            from rl_agent import PROJECT_ROOT
            model_path = PROJECT_ROOT / "models" / config_tag / "model"
        model_file = model_path.with_suffix(".zip")
        if not model_file.exists():
            print(f"Error: No model found at {model_file}.")
            print(f"  Config tag: {config_tag}")
            print(f"  Train one first with: python main.py --train --no-display "
                  f"--n-steps {args.n_steps} --batch-size {args.batch_size} "
                f"--lr {args.lr} --ent-coef {args.ent_coef}"
                  + (" --lr-decay" if args.lr_decay else ""))
            if args.original_env:
                print("  Add --original-env to run the original Gymnasium environment.")
            sys.exit(1)
        print(f"Loading model from {model_file} for evaluation ...")
        print(f"  Config: {config_tag}")
        model = load_model(model_path)

        # Load baseline model if specified
        baseline_model = None
        if baseline_tag:
            from rl_agent import PROJECT_ROOT
            baseline_path = PROJECT_ROOT / "models" / baseline_tag / "model"
            baseline_file = baseline_path.with_suffix(".zip")
            if baseline_file.exists():
                print(f"  Baseline: {baseline_tag}")
                baseline_model = load_model(baseline_path)
            else:
                print(f"  \u26a0  Baseline model not found at {baseline_file}, falling back to random baseline.")
                baseline_tag = ""

        full_evaluation(
            model=model,
            episodes=args.eval_episodes,
            seed=args.seed,
            model_dir=model_path.parent,
            config_tag=config_tag,
            use_original_env=args.original_env,
            baseline_model=baseline_model,
            baseline_tag=baseline_tag,
        )
        return

    if args.train:
        train_then_run(
            total_timesteps=args.timesteps,
            episodes=args.episodes,
            model_path=args.model_path,  # None → auto from config tag
            seed=args.seed,
            render_mode=None if args.no_display else "human",
            continue_training=args.continue_training,
            eval_episodes=args.eval_episodes,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            ent_coef=args.ent_coef,
            lr_decay=args.lr_decay,
            use_original_env=args.original_env,
            baseline_tag=baseline_tag,
        )
        return

    # Default: load and run an existing model
    if args.model_path is None:
        from rl_agent import PROJECT_ROOT
        model_path = PROJECT_ROOT / "models" / config_tag / "model"
    model_file = model_path.with_suffix(".zip")
    if not model_file.exists():
        print(f"Model not found at {model_file}. Training a new PPO agent first.")
        train_then_run(
            total_timesteps=args.timesteps,
            episodes=args.episodes,
            model_path=args.model_path,
            seed=args.seed,
            render_mode=None if args.no_display else "human",
            eval_episodes=args.eval_episodes,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            ent_coef=args.ent_coef,
            lr_decay=args.lr_decay,
            use_original_env=args.original_env,
        )
        return

    model = load_model(model_path)
    run_policy(model, episodes=args.episodes, render_mode=None if args.no_display else "human",
               seed=args.seed, config_tag=config_tag, use_original_env=args.original_env)


if __name__ == "__main__":
    main()