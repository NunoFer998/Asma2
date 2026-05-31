"""Comprehensive evaluation module for the PPO agent."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import numpy as np
from envs.finite_fuel_wrapper import FiniteFuelWrapper

from envs.custom_lunar_lander import CustomLunarLander
from gymnasium.envs.box2d.lunar_lander import LunarLander

PROJECT_ROOT = Path(__file__).resolve().parent

SOLVED_REWARD    = 200.0
SOFT_LAND_REWARD = 0.0


# ── Data collection ───────────────────────────────────────────────────

def _run_episodes(
    predict_fn,
    episodes: int,
    seed: int,
    label: str = "Agent",
    use_original_env: bool = False,
) -> list[dict]:
    """Run *episodes* evaluation episodes."""
    env = LunarLander(render_mode=None) if use_original_env else FiniteFuelWrapper(CustomLunarLander(render_mode=None))

    results: list[dict] = []

    for ep in range(episodes):
        obs, _info = env.reset(seed=seed + ep)
        terminated = truncated = False
        total_reward = 0.0
        step_count = 0
        final_obs = obs

        while not (terminated or truncated):
            action = predict_fn(obs)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            step_count += 1
            final_obs = obs

        # Classify outcome
        legs_contact = (float(final_obs[6]) >= 0.5) and (float(final_obs[7]) >= 0.5)
        landed_safely = terminated and (not truncated) and legs_contact and total_reward > SOFT_LAND_REWARD
        solved        = landed_safely and total_reward >= SOLVED_REWARD

        results.append({
            "episode":       ep + 1,
            "total_reward":  round(total_reward, 4),
            "total_steps":   step_count,
            "pad_chunk":     getattr(env.unwrapped, "_pad_idx", None),
            "terminated":    terminated,
            "truncated":     truncated,
            "legs_contact":  legs_contact,
            "landed_safely": landed_safely,
            "solved":        solved,
        })

        if (ep + 1) % 10 == 0 or (ep + 1) == episodes:
            print(f"  [{label}] {ep + 1}/{episodes} episodes completed …")

    env.close()
    return results


def run_agent_evaluation(
    model,
    episodes: int = 50,
    seed: int = 42,
    use_original_env: bool = False,
) -> list[dict]:
    """Run the trained PPO agent for *episodes* episodes."""
    print(f"\n{'─' * 55}")
    environment_label = "original environment" if use_original_env else "custom environment"
    print(f"  Running trained agent evaluation on the {environment_label} ({episodes} episodes)")
    print(f"{'─' * 55}")
    return _run_episodes(
        predict_fn=lambda obs: model.predict(obs, deterministic=True)[0],
        episodes=episodes,
        seed=seed,
        label="Agent",
        use_original_env=use_original_env,
    )


def run_random_evaluation(episodes: int = 50, seed: int = 42, use_original_env: bool = False) -> list[dict]:
    """Run the random baseline for *episodes* episodes."""
    rng = np.random.default_rng(seed)
    probe = LunarLander(render_mode=None) if use_original_env else FiniteFuelWrapper(CustomLunarLander(render_mode=None))
    n_actions = probe.action_space.n
    probe.close()

    print(f"\n{'─' * 55}")
    baseline_label = "original environment baseline" if use_original_env else "random baseline"
    print(f"  Running {baseline_label} evaluation ({episodes} episodes)")
    print(f"{'─' * 55}")
    return _run_episodes(
        predict_fn=lambda _obs: int(rng.integers(n_actions)),
        episodes=episodes,
        seed=seed,
        label="Random",
        use_original_env=use_original_env,
    )


def run_model_baseline_evaluation(
    baseline_model,
    episodes: int = 50,
    seed: int = 42,
    baseline_name: str = "Baseline Model",
    use_original_env: bool = False,
) -> list[dict]:
    """Run a previously trained model as the baseline for *episodes* episodes."""
    print(f"\n{'─' * 55}")
    print(f"  Running {baseline_name} evaluation ({episodes} episodes)")
    print(f"{'─' * 55}")
    return _run_episodes(
        predict_fn=lambda obs: baseline_model.predict(obs, deterministic=True)[0],
        episodes=episodes,
        seed=seed,
        label=baseline_name,
        use_original_env=use_original_env,
    )


# ── Metrics computation ─────────────────────────────────────────────────

def compute_metrics(
    agent_results: list[dict],
    random_results: list[dict],
) -> dict[str, Any]:
    """Compute all evaluation metrics from raw episode data."""

    def _stats(results: list[dict]) -> dict:
        rewards = [r["total_reward"] for r in results]
        steps   = [r["total_steps"]  for r in results]
        n       = len(results)

        landed = [r for r in results if r["landed_safely"]]
        solved = [r for r in results if r["solved"]]

        chunk_rewards:  dict[int, list[float]] = defaultdict(list)
        chunk_successes: dict[int, list[bool]] = defaultdict(list)
        for r in results:
            c = r["pad_chunk"]
            if c is not None:
                chunk_rewards[c].append(r["total_reward"])
                chunk_successes[c].append(r["landed_safely"])

        per_chunk = {}
        for c in sorted(chunk_rewards):
            cr = chunk_rewards[c]
            cs = chunk_successes[c]
            per_chunk[c] = {
                "episodes":     len(cr),
                "mean_reward":  round(statistics.mean(cr), 2),
                "success_rate": round(sum(cs) / len(cs) * 100, 1) if cs else 0.0,
            }

        return {
            "episodes":            n,
            "mean_reward":         round(statistics.mean(rewards), 2),
            "std_reward":          round(statistics.stdev(rewards), 2) if n > 1 else 0.0,
            "median_reward":       round(statistics.median(rewards), 2),
            "min_reward":          round(min(rewards), 2),
            "max_reward":          round(max(rewards), 2),
            "mean_steps":          round(statistics.mean(steps), 1),
            "landing_success_pct": round(len(landed) / n * 100, 1),
            "solved_pct":          round(len(solved) / n * 100, 1),
            "crash_pct":           round(sum(1 for r in results if r["terminated"] and not r["landed_safely"]) / n * 100, 1),
            "timeout_pct":         round(sum(1 for r in results if r["truncated"]) / n * 100, 1),
            "per_chunk":           per_chunk,
        }

    agent_stats  = _stats(agent_results)
    random_stats = _stats(random_results)

    random_mean = random_stats["mean_reward"]
    agent_mean  = agent_stats["mean_reward"]
    abs_improvement = round(agent_mean - random_mean, 2)

    if random_mean != 0:
        rel_improvement = round(abs_improvement / abs(random_mean), 2)
    else:
        rel_improvement = float("inf")

    return {
        "agent":              agent_stats,
        "random":             random_stats,
        "improvement_ratio":  rel_improvement,
        "improvement_abs":    abs_improvement,
    }


# ── Report generation ──────────────────────────────────────────────────

def _format_report(metrics: dict, timestamp: str, baseline_name: str = "Random Baseline") -> str:
    """Build the human-readable evaluation report text."""
    a = metrics["agent"]
    r = metrics["random"]

    lines: list[str] = []
    lines.append(f"{'═' * 60}")
    lines.append(f"  EVALUATION REPORT")
    lines.append(f"  {timestamp[:8]}  {timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}")
    lines.append(f"{'═' * 60}")
    lines.append("")
    lines.append("  1) MEAN RETURN ± STD")
    bl_label = baseline_name[:8].ljust(8) if len(baseline_name) > 8 else baseline_name.ljust(8)
    lines.append(f"     Agent    :  {a['mean_reward']:>8.2f} ± {a['std_reward']:.2f}   (median {a['median_reward']:.2f})")
    lines.append(f"     {bl_label}:  {r['mean_reward']:>8.2f} ± {r['std_reward']:.2f}   (median {r['median_reward']:.2f})")
    lines.append(f"     Range  :  [{a['min_reward']:.2f} … {a['max_reward']:.2f}]")
    lines.append("")
    lines.append("  2) LANDING SUCCESS RATE")
    lines.append(f"     Soft landing (reward > 0) :  {a['landing_success_pct']:>5.1f}%")
    lines.append(f"     Solved (reward ≥ 200)     :  {a['solved_pct']:>5.1f}%")
    lines.append(f"     Crash rate                :  {a['crash_pct']:>5.1f}%")
    lines.append(f"     Timeout rate              :  {a['timeout_pct']:>5.1f}%")
    lines.append("")
    lines.append("  3) SUCCESS RATE PER PAD CHUNK")
    lines.append(f"     {'Chunk':<8} {'Episodes':>9} {'MeanReward':>12} {'Success%':>10}")
    lines.append(f"     {'─'*8} {'─'*9} {'─'*12} {'─'*10}")
    for c, stats in sorted(a["per_chunk"].items()):
        lines.append(f"     {c:<8} {stats['episodes']:>9} {stats['mean_reward']:>12.2f} {stats['success_rate']:>9.1f}%")
    lines.append("")
    lines.append(f"  4) IMPROVEMENT OVER {baseline_name.upper()}")
    imp = metrics["improvement_ratio"]
    if isinstance(imp, float) and imp == float("inf"):
        lines.append(f"     Improvement : ∞  (baseline mean ≈ 0)")
    else:
        lines.append(f"     Improvement : {imp}×")
    lines.append(f"     Agent mean    : {a['mean_reward']:.2f}")
    lines.append(f"     Baseline mean : {r['mean_reward']:.2f}")
    lines.append("")
    lines.append("  5) ADDITIONAL STATISTICS")
    lines.append(f"     Mean episode length : {a['mean_steps']:.1f} steps")
    lines.append(f"     Evaluation episodes : {a['episodes']}")
    lines.append(f"{'═' * 60}")

    return "\n".join(lines) + "\n"


# ── Plot generation ─────────────────────────────────────────────────────

def _setup_plot_style():
    """Apply a consistent dark theme to plots."""
    plt.rcParams.update({
        "figure.facecolor":  "#1e1e2e",
        "axes.facecolor":    "#1e1e2e",
        "axes.edgecolor":    "#45475a",
        "axes.labelcolor":   "#cdd6f4",
        "text.color":        "#cdd6f4",
        "xtick.color":       "#a6adc8",
        "ytick.color":       "#a6adc8",
        "grid.color":        "#313244",
        "grid.alpha":        0.6,
        "font.size":         11,
        "axes.titlesize":    14,
        "axes.titleweight":  "bold",
        "figure.titlesize":  16,
        "figure.titleweight":"bold",
    })


def plot_reward_distribution(
    agent_results: list[dict],
    random_results: list[dict],
    save_path: Path,
    baseline_name: str = "Random Baseline",
) -> Path:
    """Side-by-side reward histograms for agent vs baseline."""
    _setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    agent_rewards  = [r["total_reward"] for r in agent_results]
    random_rewards = [r["total_reward"] for r in random_results]
    axes[0].hist(agent_rewards, bins=20, color="#89b4fa", edgecolor="#1e1e2e", alpha=0.9)
    axes[0].axvline(statistics.mean(agent_rewards), color="#f38ba8", linestyle="--", linewidth=2, label=f"Mean: {statistics.mean(agent_rewards):.1f}")
    axes[0].axvline(200, color="#a6e3a1", linestyle=":", linewidth=2, label="Solved (200)")
    axes[0].set_title("Trained Agent")
    axes[0].set_xlabel("Episode Return")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, axis="y")
    axes[1].hist(random_rewards, bins=20, color="#fab387", edgecolor="#1e1e2e", alpha=0.9)
    axes[1].axvline(statistics.mean(random_rewards), color="#f38ba8", linestyle="--", linewidth=2, label=f"Mean: {statistics.mean(random_rewards):.1f}")
    axes[1].axvline(200, color="#a6e3a1", linestyle=":", linewidth=2, label="Solved (200)")
    axes[1].set_title(baseline_name)
    axes[1].set_xlabel("Episode Return")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, axis="y")

    fig.suptitle(f"Reward Distribution — Agent vs {baseline_name}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {save_path}")
    return save_path


def plot_success_per_chunk(
    agent_results: list[dict],
    save_path: Path,
) -> Path:
    """Bar chart of success rate and mean reward per landing-pad chunk."""
    _setup_plot_style()

    chunk_data: dict[int, dict] = defaultdict(lambda: {"rewards": [], "successes": []})
    for r in agent_results:
        c = r["pad_chunk"]
        if c is not None:
            chunk_data[c]["rewards"].append(r["total_reward"])
            chunk_data[c]["successes"].append(r["landed_safely"])

    chunks  = sorted(chunk_data)
    means   = [statistics.mean(chunk_data[c]["rewards"]) for c in chunks]
    success = [sum(chunk_data[c]["successes"]) / len(chunk_data[c]["successes"]) * 100 for c in chunks]
    counts  = [len(chunk_data[c]["rewards"]) for c in chunks]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    x = np.arange(len(chunks))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, means, width, color="#89b4fa", alpha=0.85, label="Mean Reward")
    ax1.set_ylabel("Mean Reward")
    ax1.set_xlabel("Pad Chunk Index")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Chunk {c}\n(n={counts[i]})" for i, c in enumerate(chunks)])
    ax1.axhline(200, color="#a6e3a1", linestyle=":", linewidth=1.5, alpha=0.7, label="Solved (200)")
    ax1.grid(True, axis="y")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, success, width, color="#f9e2af", alpha=0.85, label="Success %")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_ylim(0, 110)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=9)

    fig.suptitle("Performance per Landing Pad Position (Generalisation)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {save_path}")
    return save_path


def plot_learning_curve(
    eval_log_path: Path,
    save_path: Path,
) -> Path | None:
    """Plot the learning curve from the EvalCallback NPZ log."""
    npz_file = eval_log_path / "evaluations.npz"
    if not npz_file.exists():
        print(f"  ⚠  EvalCallback log not found at {npz_file}, skipping learning curve.")
        return None

    _setup_plot_style()
    data = np.load(str(npz_file))

    timesteps = data["timesteps"]
    results   = data["results"]

    mean_rewards = results.mean(axis=1)
    std_rewards  = results.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(timesteps, mean_rewards, color="#89b4fa", linewidth=2, label="Mean Eval Reward")
    ax.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        color="#89b4fa",
        alpha=0.2,
        label="± 1 Std Dev",
    )
    ax.axhline(200, color="#a6e3a1", linestyle=":", linewidth=1.5, alpha=0.7, label="Solved (200)")
    ax.axhline(0,   color="#f38ba8", linestyle=":", linewidth=1,   alpha=0.5, label="Zero Reward")

    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Eval Reward")
    ax.set_title("Learning Curve (during training)")
    ax.legend(fontsize=9)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {save_path}")
    return save_path


# ── Main entry point ────────────────────────────────────────────────────

def full_evaluation(
    model,
    episodes: int = 50,
    seed: int = 42,
    model_dir: Path | None = None,
    config_tag: str = "",
    use_original_env: bool = False,
    baseline_model=None,
    baseline_tag: str = "",
) -> dict[str, Any]:
    """Run the complete evaluation pipeline."""
    model_dir = model_dir or (PROJECT_ROOT / "models")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if baseline_model is not None:
        baseline_name = f"Baseline: {baseline_tag}" if baseline_tag else "Baseline Model"
    elif use_original_env:
        baseline_name = "Original Environment Baseline"
    else:
        baseline_name = "Random Baseline"

    agent_results = run_agent_evaluation(model, episodes=episodes, seed=seed, use_original_env=use_original_env)

    if baseline_model is not None:
        random_results = run_model_baseline_evaluation(
            baseline_model,
            episodes=episodes,
            seed=seed + 1000,
            baseline_name=baseline_name,
            use_original_env=use_original_env,
        )
    else:
        random_results = run_random_evaluation(episodes=episodes, seed=seed + 1000, use_original_env=use_original_env)

    # Compute metrics
    metrics = compute_metrics(agent_results, random_results)

    # Generate & save report
    report_text = _format_report(metrics, timestamp, baseline_name=baseline_name)
    if config_tag:
        report_text = f"  Config: {config_tag}\n" + report_text
    print(f"\n{report_text}")

    log_dir = PROJECT_ROOT / "logs"
    if config_tag:
        log_dir = log_dir / config_tag
    log_dir.mkdir(parents=True, exist_ok=True)

    report_file = log_dir / f"evaluation_report_{timestamp}.txt"
    report_file.write_text(report_text, encoding="utf-8")

    print(f"Report saved → {report_file}")

    json_file = log_dir / f"evaluation_report_{timestamp}.json"
    output_metrics = {**metrics, "config_tag": config_tag} if config_tag else metrics
    json_file.write_text(json.dumps(output_metrics, indent=2),encoding="utf-8")
    print(f"JSON   saved → {json_file}")

    # Generate plots
    plot_dir = log_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots …")
    plot_reward_distribution(
        agent_results, random_results,
        plot_dir / f"reward_distribution_{timestamp}.png",
        baseline_name=baseline_name,
    )
    plot_success_per_chunk(
        agent_results,
        plot_dir / f"success_per_chunk_{timestamp}.png",
    )
    plot_learning_curve(
        model_dir / "eval_logs",
        plot_dir / f"learning_curve_{timestamp}.png",
    )

    print(f"\n{'═' * 55}")
    print(f"  ✓ Full evaluation complete — {episodes} episodes")
    print(f"{'═' * 55}\n")

    return metrics
