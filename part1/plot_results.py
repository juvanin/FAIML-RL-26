"""Plot learning curves and produce a summary table from training results.

Expected directory layout:
    results/
      <algorithm>/
        seed_<N>/
          log.csv
          eval_log.csv

Usage:
    python plot_results.py
    python plot_results.py --results-dir results --window 50
    python plot_results.py --algorithms reinforce ac_td
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------

COLORS = {
    'reinforce':        '#e74c3c',
    'reinforce_batch':  '#3498db',
    'reinforce_ema':    '#2ecc71',
    'reinforce_fixed':  '#1abc9c',
    'ac_mc':            '#9b59b6',
    'ac_td':            '#f39c12',
}

LABELS = {
    'reinforce':        'REINFORCE',
    'reinforce_batch':  'REINFORCE (batch baseline)',
    'reinforce_ema':    'REINFORCE (EMA baseline)',
    'reinforce_fixed':  'REINFORCE (fixed baseline)',
    'ac_mc':            'Actor-Critic (MC)',
    'ac_td':            'Actor-Critic (TD)',
}


def load_csv(path):
    """Load a CSV file into a dict of column-lists."""
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        data = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k, v in row.items():
                if v == '':
                    data[k].append(np.nan)
                else:
                    try:
                        data[k].append(float(v))
                    except ValueError:
                        data[k].append(v)
    return data


def smooth(values, window=50):
    """Rolling-mean smoothing with edge-padding."""
    n = len(values)
    if n == 0:
        return values
    w = min(window, n)
    kernel = np.ones(w) / w
    padded = np.pad(values, (w - 1, 0), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plot training results for Hopper-v4 experiments')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Base results directory')
    parser.add_argument('--window', type=int, default=50,
                        help='Smoothing window for learning curves')
    parser.add_argument('--algorithms', nargs='+',
                        default=['reinforce', 'reinforce_batch',
                                 'reinforce_ema', 'reinforce_fixed',
                                 'ac_mc', 'ac_td'],
                        help='Algorithms to include in the plots')
    parser.add_argument('--conv-threshold', type=float, default=100.0,
                        help='Return threshold to consider as converged')
    args = parser.parse_args()

    plot_dir = os.path.join(args.results_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # ==================================================================
    #  1.  Learning curves  (training return vs. episode)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    summary_rows = []

    for algo in args.algorithms:
        seed_dirs = sorted(
            glob.glob(os.path.join(args.results_dir, algo, 'seed_*')))
        if not seed_dirs:
            print(f"[skip] no data for '{algo}'")
            continue

        all_returns = []
        wall_times = []
        for sd in seed_dirs:
            log_path = os.path.join(sd, 'log.csv')
            if not os.path.exists(log_path):
                continue
            data = load_csv(log_path)
            returns = np.array(data['return'], dtype=float)
            all_returns.append(returns)
            wt = [x for x in data.get('wall_time', []) if not np.isnan(x)]
            if wt:
                wall_times.append(wt[-1])

        if not all_returns:
            continue

        # Truncate to shortest seed
        min_len = min(len(r) for r in all_returns)
        all_returns = np.array([r[:min_len] for r in all_returns])

        # Smooth per seed, then compute mean ± std across seeds
        smoothed = np.array([smooth(r, args.window) for r in all_returns])
        mean_curve = smoothed.mean(axis=0)
        std_curve = smoothed.std(axis=0)
        episodes = np.arange(1, min_len + 1)

        color = COLORS.get(algo, 'gray')
        label = LABELS.get(algo, algo)
        ax.plot(episodes, mean_curve, color=color, label=label, linewidth=2)
        ax.fill_between(episodes, mean_curve - std_curve,
                        mean_curve + std_curve, color=color, alpha=0.15)

        # Summary statistics
        mean_returns = all_returns.mean(axis=0)
        mean100 = mean_returns[-100:].mean()
        std100  = mean_returns[-100:].std()
        mean500 = mean_returns[-500:].mean() if len(mean_returns) >= 500 else mean_returns.mean()
        std500  = mean_returns[-500:].std()  if len(mean_returns) >= 500 else mean_returns.std()
        rolling = smooth(mean_returns, args.window)
        conv_ep = (int(np.argmax(rolling > args.conv_threshold) + 1)
                   if np.any(rolling > args.conv_threshold) else None)
        avg_wall = float(np.mean(wall_times)) if wall_times else None

        summary_rows.append({
            'Algorithm': label,
            'Mean (last 100)': f'{mean100:.1f}',
            'Std (last 100)':  f'{std100:.1f}',
            'Mean (last 500)': f'{mean500:.1f}',
            'Std (last 500)':  f'{std500:.1f}',
            f'Convergence Ep (>{args.conv_threshold:g})': str(conv_ep) if conv_ep else 'N/A',
            'Wall Time (s)': f'{avg_wall:.0f}' if avg_wall else 'N/A',
        })

    ax.set_xlabel('Episode', fontsize=13)
    ax.set_ylabel('Episode Return (smoothed)', fontsize=13)
    ax.set_title('Hopper-v4: Learning Curves', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    lc_path = os.path.join(plot_dir, 'learning_curves.png')
    fig.savefig(lc_path, dpi=150)
    print(f"Saved {lc_path}")
    plt.close(fig)

    # ==================================================================
    #  2.  Evaluation curves  (eval mean return vs. episode checkpoint)
    # ==================================================================
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    has_eval = False

    for algo in args.algorithms:
        seed_dirs = sorted(
            glob.glob(os.path.join(args.results_dir, algo, 'seed_*')))
        if not seed_dirs:
            continue

        all_eval_returns, all_eval_eps = [], []
        for sd in seed_dirs:
            eval_path = os.path.join(sd, 'eval_log.csv')
            if not os.path.exists(eval_path):
                continue
            data = load_csv(eval_path)
            if data.get('eval_mean_return'):
                all_eval_returns.append(
                    np.array(data['eval_mean_return'], dtype=float))
                all_eval_eps.append(
                    np.array(data['episode_checkpoint'], dtype=float))

        if not all_eval_returns:
            continue

        has_eval = True
        min_len = min(len(r) for r in all_eval_returns)
        eval_returns = np.array([r[:min_len] for r in all_eval_returns])
        eval_eps = all_eval_eps[0][:min_len]

        mean_eval = eval_returns.mean(axis=0)
        std_eval = eval_returns.std(axis=0)

        color = COLORS.get(algo, 'gray')
        label = LABELS.get(algo, algo)
        ax2.plot(eval_eps, mean_eval, color=color, label=label,
                 linewidth=2, marker='o', markersize=3)
        ax2.fill_between(eval_eps, mean_eval - std_eval,
                         mean_eval + std_eval, color=color, alpha=0.15)

    if has_eval:
        ax2.set_xlabel('Episode', fontsize=13)
        ax2.set_ylabel('Eval Mean Return', fontsize=13)
        ax2.set_title('Hopper-v4: Evaluation Curves', fontsize=15)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        ec_path = os.path.join(plot_dir, 'eval_curves.png')
        fig2.savefig(ec_path, dpi=150)
        print(f"Saved {ec_path}")
    plt.close(fig2)

    # ==================================================================
    #  3.  Summary table (printed to console)
    # ==================================================================
    if summary_rows:
        print(f"\n{'='*80}")
        print("SUMMARY TABLE")
        print(f"{'='*80}")
        headers = list(summary_rows[0].keys())
        widths = {h: max(len(h), max(len(r[h]) for r in summary_rows))
                  for h in headers}
        hdr = ' | '.join(h.ljust(widths[h]) for h in headers)
        print(hdr)
        print('-' * len(hdr))
        for row in summary_rows:
            print(' | '.join(
                str(row[h]).ljust(widths[h]) for h in headers))
        print()


if __name__ == '__main__':
    main()
