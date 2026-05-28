# Custom Lunar Lander — PPO Agent

A reinforcement learning agent trained with **PPO** (Proximal Policy Optimisation) to land on a **randomised landing pad** in a modified Gymnasium LunarLander environment.

## Key Features

- **Random landing pad** — the pad spawns in a different position each episode (chunks 2–8), forcing the agent to generalise
- **Corrective reward shaping** — fixes the parent environment's shaping signal to point at the actual pad, not the screen centre
- **Precision landing bonus** — Gaussian-shaped reward for landing close to the pad centre
- **4 parallel training environments** with different seeds for diverse experience
- **Automatic evaluation** — 50-episode benchmark with reports and plots after training
- **Hyperparameter benchmarking** — configurable via CLI with auto-organised output folders

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- `gymnasium[box2d]`, `pygame`, `stable-baselines3>=2.3.0`, `matplotlib`, `tensorboard`

---

## Quick Start

```bash
# Train with default hyperparameters (500k timesteps, headless)
python main.py --train --no-display

# Run a pre-trained model (opens a window)
python main.py

# Run a random baseline for comparison
python main.py --random --no-display

# Run with the original Gymnasium LunarLander environment
python main.py --train --no-display --original-env
```

---

## CLI Reference

### Mode Flags

| Flag | Description |
|------|-------------|
| `--train` | Train a new PPO agent, then run demo episodes + automatic evaluation |
| `--evaluate` | Run a comprehensive 50-episode evaluation on an existing model |
| `--random` | Run a random-action baseline (for comparison) |
| `--original-env` | Run against the original Gymnasium LunarLander environment instead of the custom wrapper |
| *(no flag)* | Load and run an existing trained model |

### General Options

| Flag | Default | Description |
|------|---------|-------------|
| `--timesteps N` | `500000` | Total training timesteps |
| `--episodes N` | `5` | Number of demo episodes to render after training |
| `--seed N` | `42` | Random seed for reproducibility |
| `--model-path PATH` | *(auto)* | Explicit path to save/load the model (overrides auto-naming) |
| `--no-display` | off | Run headless (no render window) — use for remote servers |
| `--continue` | off | Continue training from an existing model instead of starting fresh |
| `--eval-episodes N` | `50` | Number of episodes for the comprehensive evaluation |

### Hyperparameters (for benchmarking)

| Flag | Default | Suggested Values | Description |
|------|---------|-----------------|-------------|
| `--n-steps N` | `2048` | 1024, 2048, 4096 | Rollout length per env per PPO update |
| `--batch-size N` | `64` | 64, 128, 256 | Minibatch size for SGD |
| `--lr FLOAT` | `3e-4` | 1e-4, 3e-4, 1e-3 | Learning rate for Adam optimiser |
| `--ent-coef FLOAT` | `0.01` | 0.0, 0.01, 0.05 | Entropy coefficient (exploration) |
| `--lr-decay` | off | — | Linear LR decay from `--lr` → 0 over training |

When `--original-env` is set, the run uses the stock LunarLander environment and writes outputs under an `orig_...` config tag so it stays separate from the custom-environment runs.

---

## Usage Examples

### Training

```bash
# Default config
python main.py --train --no-display

# Custom hyperparameters
python main.py --train --no-display --n-steps 1024 --batch-size 128 --lr 1e-3 --ent-coef 0.05

# With linear learning rate decay
python main.py --train --no-display --lr 3e-4 --lr-decay

# Longer training
python main.py --train --no-display --timesteps 1000000

# Continue training an existing model for more timesteps
python main.py --train --no-display --continue --timesteps 200000
```

### Evaluation

```bash
# Evaluate the default config model
python main.py --evaluate --no-display

# Evaluate a specific config
python main.py --evaluate --no-display --n-steps 1024 --batch-size 128 --lr 1e-3 --ent-coef 0.05

# Evaluate with more episodes for higher statistical confidence
python main.py --evaluate --no-display --eval-episodes 100

# Evaluate a model at a specific path
python main.py --evaluate --model-path models/ns1024_bs128_lr1e-03_ec0.05/model
```

### Comparing Runs with TensorBoard

```bash
# Launch TensorBoard to compare all training runs side-by-side
tensorboard --logdir tensorboard/
```

---

## Hyperparameter Benchmarking

Each combination of hyperparameters generates a unique **config tag** (e.g. `ns1024_bs128_lr1e-03_ec0.05_decay`) that is used to organise all outputs automatically:

```
models/
├── ns2048_bs64_lr3e-04_ec0.01/          # Default config
│   ├── model.zip                        # Trained model
│   ├── best_model/                      # Best checkpoint (by eval reward)
│   ├── checkpoints/                     # Periodic checkpoints
│   └── eval_logs/evaluations.npz        # EvalCallback data (learning curve)
├── ns1024_bs128_lr1e-03_ec0.05/         # Custom config
│   └── ...
├── ns2048_bs64_lr3e-04_ec0.01_decay/    # With LR decay
│   └── ...

logs/
├── ns2048_bs64_lr3e-04_ec0.01/          # Matching eval outputs
│   ├── evaluation_report_*.txt          # Human-readable report
│   ├── evaluation_report_*.json         # Machine-readable metrics
│   ├── episode_actions_*.json           # Step-by-step episode data
│   └── plots/
│       ├── reward_distribution_*.png    # Agent vs Random histogram
│       ├── success_per_chunk_*.png      # Per-pad-position performance
│       └── learning_curve_*.png         # Reward over training timesteps

tensorboard/
├── ns2048_bs64_lr3e-04_ec0.01/          # TensorBoard logs per config
├── ns1024_bs128_lr1e-03_ec0.05/
└── ...
```

### Example: Run a Grid Search

```bash
# Sweep over learning rates
for lr in 1e-4 3e-4 1e-3; do
    python main.py --train --no-display --lr $lr --timesteps 500000
done

# Sweep with LR decay
python main.py --train --no-display --lr 3e-4 --lr-decay

# Compare all runs
tensorboard --logdir tensorboard/
```

---

## Automatic Evaluation Report

After training (or with `--evaluate`), a **50-episode evaluation** runs automatically and produces:

1. **Mean Return ± Std** — primary performance metric
2. **Landing Success Rate (%)** — soft landing + solved (reward ≥ 200)
3. **Success Rate per Pad Chunk** — proves the agent generalises across pad positions
4. **Improvement over Random Baseline** — quantifies how much better the agent is
5. **Learning Curve** — reward over training timesteps (from TensorBoard data)

Example output:

```
════════════════════════════════════════════════════════════
  EVALUATION REPORT
════════════════════════════════════════════════════════════

  1) MEAN RETURN ± STD
     Agent  :    245.12 ± 38.50   (median 258.30)
     Random :   -189.80 ± 96.12   (median -158.90)

  2) LANDING SUCCESS RATE
     Soft landing (reward > 0) :   92.0%
     Solved (reward ≥ 200)     :   78.0%

  3) SUCCESS RATE PER PAD CHUNK
     Chunk     Episodes   MeanReward   Success%
     2                5       238.10      80.0%
     3                8       252.30      87.5%
     4               10       261.40      90.0%
     ...

  4) IMPROVEMENT OVER RANDOM BASELINE
     Improvement : 2.29×
```

---

## Custom Environment

The [`CustomLunarLander`](envs/custom_lunar_lander.py) extends Gymnasium's `LunarLander-v2` with:

| Feature | Description |
|---------|-------------|
| **Random pad position** | Landing pad spawns in chunk 2–8 each episode |
| **10-dim observation** | Original 8 dims + normalised pad x/y position |
| **Corrective shaping** | Fixes parent's reward to point at pad, not screen centre |
| **Precision bonus** | +40 Gaussian bonus for landing exactly on pad centre |
| **Robust crash detection** | Uses `game_over` physics flag instead of reward threshold |

### Observation Space (10 dimensions)

| Index | Description |
|-------|-------------|
| 0 | x position relative to **pad** (not screen centre) |
| 1 | y position relative to pad |
| 2 | x velocity |
| 3 | y velocity |
| 4 | Lander angle |
| 5 | Angular velocity |
| 6 | Left leg contact (0 or 1) |
| 7 | Right leg contact (0 or 1) |
| 8 | Pad x position (normalised, -1 to 1) |
| 9 | Pad y position (normalised) |
