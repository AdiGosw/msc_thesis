import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, ttest_1samp
from scipy.optimize import brentq, minimize_scalar

SIGMA   = 5
DELTA_V = 4
N       = 2
E_V     = 2.0

unibonn_blue = '#004E9F'
unibonn_yellow = '#FCBA00'
unibonn_grey   = '#909085'
THESIS_DIR   = 'C:/Users/adigo/Desktop/Thesis folder/Masters_Thesis'


# ── File cache (each .npy loaded at most once per session) ────────────────────

_DATA_CACHE = {}

def _load(path):
    if path not in _DATA_CACHE:
        _DATA_CACHE[path] = np.load(path, allow_pickle=True)
    return _DATA_CACHE[path]


# ── Vectorised smoother ────────────────────────────────────────────────────────

def smooth(arr, w):
    """Causal expanding-window rolling mean — fully vectorised via cumsum."""
    arr    = np.asarray(arr, dtype=float)
    cumsum = np.cumsum(np.concatenate([[0.0], arr]))
    idx    = np.arange(len(arr))
    start  = np.maximum(0, idx - w + 1)
    return (cumsum[idx + 1] - cumsum[start]) / (idx - start + 1)


# ── Loaders ───────────────────────────────────────────────────────────────────

SIGMAS = [1, 3, 5, 7, 9]

def load_ddpg(sigma=SIGMA):
    data = _load(f"results_sigma{sigma}_N{N}.npy")
    if data.ndim == 0:
        return data.item()
    runs = list(data)
    combined = dict(runs[0])
    combined['quoted_spreads']   = [s for r in runs for s in r['quoted_spreads']]
    combined['realized_spreads'] = [s for r in runs for s in r['realized_spreads']]
    return combined

def load_ddpg_nas(sigma=SIGMA):
    data = _load(f"results_nas_sigma{sigma}_N{N}.npy")
    if data.ndim == 0:
        return data.item()
    runs = list(data)
    combined = dict(runs[0])
    combined['quoted_spreads']   = [s for r in runs for s in r['quoted_spreads']]
    combined['realized_spreads'] = [s for r in runs for s in r['realized_spreads']]
    return combined

def load_ql(sigma=SIGMA):
    return _load(f"qlearning_sigma{sigma}_N2.npy").item()

def load_ql_nas(sigma=SIGMA):
    return _load(f"qlearning_sigma{sigma}_N2_nas.npy").item()

def _ddpg_run_means(sigma, key, nas=False):
    prefix = "results_nas_sigma" if nas else "results_sigma"
    data = _load(f"{prefix}{sigma}_N{N}.npy")
    if data.ndim == 0:
        return [float(np.mean(data.item()[key]))]
    return [float(np.mean(r[key])) for r in list(data)]

def _ddpg_run_means_N(n_agents, key, sigma=5, nas=False):
    prefix = "results_nas_sigma" if nas else "results_sigma"
    data = _load(f"{prefix}{sigma}_N{n_agents}.npy")
    if data.ndim == 0:
        return [float(np.mean(data.item()[key]))]
    return [float(np.mean(r[key])) for r in list(data)]

def _ql_spread_N(n_agents, key, sigma=5, nas=False):
    suffix = '_nas' if nas else ''
    data = _load(f"qlearning_sigma{sigma}_N{n_agents}{suffix}.npy").item()
    return list(data[key])


# ── Nash benchmark ─────────────────────────────────────────────────────────────

def gm_equation(a, v_high=4, v_low=0, mu=0.5, sigma=SIGMA):
    D_high = 1 - norm.cdf(a - v_high, 0, sigma)
    D_low  = 1 - norm.cdf(a - v_low,  0, sigma)
    p_buy  = mu * D_high + (1 - mu) * D_low
    return a - (E_V + (1 - mu) * mu * (v_high - v_low) * (D_high - D_low) / p_buy)

gm_price = brentq(gm_equation, E_V, 4.0)
gm_qs    = gm_price - E_V
gm_rs    = 0.0


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_price_distribution(sigma=SIGMA, agent=1):
    """
    Histogram of all greedy price snapshots across all training iterations,
    pooled across all experiments/runs, for the specified agent (1 or 2).
    Q-Learning: all price_history entries pooled across all experiments.
    DDPG: all episode_prices pooled across 10 runs.
    Y-axis: percentage of total observations. Nash shown as a vertical dotted line.
    """
    data_ddpg = np.load(f"results_sigma{sigma}_N{N}.npy", allow_pickle=True)
    runs_ddpg = list(data_ddpg) if data_ddpg.ndim != 0 else [data_ddpg.item()]
    ql = load_ql(sigma)

    nash = brentq(gm_equation, E_V, 8.0, args=(4, 0, 0.5, sigma))

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Q-Learning: all snapshots per experiment, pooled ──
    prices_ql = np.array([
        snap[f'greedy_price_{agent}']
        for ph in ql['price_history']
        for snap in ph
    ])
    mean_ql   = np.mean(prices_ql)
    median_ql = np.median(prices_ql)
    std_ql    = np.std(prices_ql, ddof=1)
    weights_ql = np.ones(len(prices_ql)) / len(prices_ql) * 100
    ax1.hist(prices_ql, bins=30, weights=weights_ql,
             color=unibonn_blue, alpha=0.8, edgecolor='black')
    ax1.axvline(nash, color='black', linestyle='--', linewidth=1.5,
                label=f'Glosten-Milgrom Price ({nash:.2f})')
    ax1.axvline(mean_ql, color=unibonn_yellow, linestyle='-', linewidth=1.5,
                label=f'Mean ({mean_ql:.2f})')
    ax1.set_title(f'Panel A: Q-Learning — Agent {agent} Price Distribution',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Price', fontsize=11)
    ax1.set_ylabel('Frequency (%)', fontsize=11)
    ax1.set_xlim(1.1, 14.9)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ── DDPG: all episode_prices per run, pooled ──
    prices_ddpg = np.array([
        p
        for run in runs_ddpg
        for p in run[f'agent{agent}']['episode_prices']
    ])
    mean_ddpg   = np.mean(prices_ddpg)
    median_ddpg = np.median(prices_ddpg)
    std_ddpg    = np.std(prices_ddpg, ddof=1)
    weights_ddpg = np.ones(len(prices_ddpg)) / len(prices_ddpg) * 100
    ax2.hist(prices_ddpg, bins=30, weights=weights_ddpg,
             color=unibonn_blue, alpha=0.8, edgecolor='black')
    ax2.axvline(nash, color='black', linestyle='--', linewidth=1.5,
                label=f'Glosten-Milgrom Price ({nash:.2f})')
    ax2.axvline(mean_ddpg, color=unibonn_yellow, linestyle='-', linewidth=1.5,
                label=f'Mean ({mean_ddpg:.2f})')
    ax2.set_title(f'Panel B: DDPG — Agent {agent} Price Distribution',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Price', fontsize=11)
    ax2.set_ylabel('Frequency (%)', fontsize=11)
    ax2.set_xlim(1.1, 14.9)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    print(f"\nPrice Distribution Statistics — Agent {agent} (sigma={sigma})")
    print(f"{'':>10} {'Mean':>8} {'Median':>8} {'Std':>8}")
    print(f"{'Q-Learning':>10} {mean_ql:>8.3f} {median_ql:>8.3f} {std_ql:>8.3f}")
    print(f"{'DDPG':>10} {mean_ddpg:>8.3f} {median_ddpg:>8.3f} {std_ddpg:>8.3f}")

    y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(0, y_max)
    ax2.set_ylim(0, y_max)

    plt.tight_layout()
    plt.savefig(f'{THESIS_DIR}/compare_price_dist_agent{agent}_sigma{sigma}.pdf',
                bbox_inches='tight', dpi=300)
    plt.show()


# def plot_price_distribution_kde(sigma=SIGMA):
#     """
#     Option 1: KDE over the final greedy price a*_{1,T} from each experiment.
#     One value per experiment (10 total), smoothed with a kernel density estimate.
#     """
#     data_ddpg = np.load(f"results_sigma{sigma}_N{N}.npy", allow_pickle=True)
#     runs_ddpg = list(data_ddpg) if data_ddpg.ndim != 0 else [data_ddpg.item()]
#     ql = load_ql(sigma)

#     nash = brentq(gm_equation, E_V, 8.0, args=(4, 0, 0.5, sigma))
#     price_range = np.linspace(1.1, 14.9, 500)

#     # Final greedy price per experiment
#     ql_final   = np.array(ql['final_prices_1'])
#     ddpg_final = np.array([run['agent1']['episode_prices'][-1] for run in runs_ddpg])

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

#     for ax, prices, title in [
#         (ax1, ql_final,   'Panel A: Q-Learning — Agent 1 Final Price Distribution'),
#         (ax2, ddpg_final, 'Panel B: DDPG — Agent 1 Final Price Distribution'),
#     ]:
#         kde = gaussian_kde(prices, bw_method='silverman')
#         ax.plot(price_range, kde(price_range), color=unibonn_blue, linewidth=2)
#         ax.fill_between(price_range, kde(price_range), alpha=0.3, color=unibonn_blue)
#         ax.axvline(nash, color='black', linestyle='--', linewidth=1.5,
#                    label=f'Nash price ({nash:.2f})')
#         # Rug plot to show individual experiment outcomes
#         ax.plot(prices, np.zeros_like(prices) - 0.002, '|', color=unibonn_yellow,
#                 markersize=12, markeredgewidth=2, label='Experiment outcomes')
#         ax.set_title(title, fontsize=12, fontweight='bold')
#         ax.set_xlabel('Price', fontsize=11)
#         ax.set_ylabel('Density', fontsize=11)
#         ax.set_xlim(1.1, 14.9)
#         ax.set_ylim(bottom=-0.01)
#         ax.legend(fontsize=10)
#         ax.grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig(f'{THESIS_DIR}/compare_price_dist_kde_sigma{sigma}.pdf',
#                 bbox_inches='tight', dpi=300)
#     plt.show()


# def plot_price_distribution_last_snapshots(sigma=SIGMA, last_n=20):
#     """
#     Option 2: Histogram over the last `last_n` greedy price snapshots per experiment.
#     last_n x 10 experiments = 200 points, capturing converged behavior only.
#     """
#     data_ddpg = np.load(f"results_sigma{sigma}_N{N}.npy", allow_pickle=True)
#     runs_ddpg = list(data_ddpg) if data_ddpg.ndim != 0 else [data_ddpg.item()]
#     ql = load_ql(sigma)

#     nash = brentq(gm_equation, E_V, 8.0, args=(4, 0, 0.5, sigma))

#     # Last last_n snapshots per experiment
#     ql_prices   = [snap['greedy_price_1']
#                    for ph in ql['price_history']
#                    for snap in ph[-last_n:]]
#     ddpg_prices = [p
#                    for run in runs_ddpg
#                    for p in run['agent1']['episode_prices'][-last_n:]]

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

#     for ax, prices, title in [
#         (ax1, ql_prices,   'Panel A: Q-Learning — Agent 1 Price Distribution (last 20 snapshots)'),
#         (ax2, ddpg_prices, 'Panel B: DDPG — Agent 1 Price Distribution (last 20 episodes)'),
#     ]:
#         total = len(prices)
#         weights = np.ones(total) / total * 100
#         ax.hist(prices, bins=30, weights=weights,
#                 alpha=0.8, edgecolor='black', color=unibonn_blue)
#         ax.axvline(nash, color='black', linestyle='--', linewidth=1.5,
#                    label=f'Nash price ({nash:.2f})')
#         ax.set_title(title, fontsize=12, fontweight='bold')
#         ax.set_xlabel('Price', fontsize=11)
#         ax.set_ylabel('Percentage of snapshots (%)', fontsize=11)
#         ax.set_xlim(1.1, 14.9)
#         ax.legend(fontsize=10)
#         ax.grid(True, alpha=0.3)

#     y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
#     ax1.set_ylim(0, y_max)
#     ax2.set_ylim(0, y_max)

#     plt.tight_layout()
#     plt.savefig(f'{THESIS_DIR}/compare_price_dist_last{last_n}_sigma{sigma}.pdf',
#                 bbox_inches='tight', dpi=300)
#     plt.show()


def compare_price_evolution(sigma=SIGMA, agent=1):
    """
    Left:  Q-Learning — mean greedy price across experiments, ± 1 std across experiments
    Right: DDPG       — mean episode price across runs, ± 1 std across runs
    Both panels show the Nash and monopoly price benchmarks as horizontal dotted lines.
    agent: which agent to plot (1 or 2).
    """
    ql = load_ql(sigma)

    # Load all DDPG runs
    data_ddpg = np.load(f"results_sigma{sigma}_N{N}.npy", allow_pickle=True)
    runs_ddpg = list(data_ddpg) if data_ddpg.ndim != 0 else [data_ddpg.item()]

    # Nash benchmark for this sigma
    nash = brentq(gm_equation, E_V, 8.0, args=(4, 0, 0.5, sigma))
    monopoly_price = minimize_scalar(
        lambda a: -(0.5*(a - 4)*(1 - norm.cdf(a - 4, 0, sigma))
                  + 0.5*a*(1 - norm.cdf(a, 0, sigma))),
        bounds=(E_V, 14.9), method='bounded').x

    # Pre-compute DDPG x-axis so both panels share the same xlim
    STEPS_PER_EPISODE = 100
    n_ep       = len(runs_ddpg[0][f'agent{agent}']['episode_prices'])
    iters_ddpg = np.arange(1, n_ep + 1) * STEPS_PER_EPISODE

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Q-Learning: mean and std across experiments ──
    price_histories = ql['price_history']
    max_length = max(len(ph) for ph in price_histories if ph)
    all_min_prices = []
    steps_ql = []
    last_known_price = {i: None for i in range(len(price_histories))}

    for i in range(max_length):
        step_prices = []
        current_step = None
        for j, ph in enumerate(price_histories):
            if ph and i < len(ph):
                price = ph[i][f'greedy_price_{agent}']
                last_known_price[j] = price
                step_prices.append(price)
                current_step = ph[i]['step']
            elif last_known_price[j] is not None:
                step_prices.append(last_known_price[j])
                if current_step is None and ph:
                    current_step = ph[-1]['step']
        if step_prices and current_step is not None:
            all_min_prices.append(step_prices)
            steps_ql.append(current_step)

    mean_prices = np.array([np.mean(p) for p in all_min_prices])
    std_prices  = np.array([np.std(p)  for p in all_min_prices])
    steps_ql    = np.array(steps_ql)

    w_ql    = max(1, len(mean_prices) // 10)
    mean_sm = smooth(mean_prices, w_ql)
    std_sm  = smooth(std_prices,  w_ql)

    # Training region
    ax1.fill_between(steps_ql, mean_sm - std_sm, mean_sm + std_sm,
                     color=unibonn_grey, alpha=0.7, label='± 1 std.')
    ax1.plot(steps_ql, mean_sm, color=unibonn_yellow, linewidth=2, label='Average')

    ax1.axhline(nash, color='black', linestyle='--', linewidth=1.5,
                label=f'Glosten-Milgrom Price ({nash:.2f})')
    ax1.axhline(monopoly_price, color=unibonn_grey, linestyle='--', linewidth=1.5,
                label=f'Monopoly Price ({monopoly_price:.2f})')
    ax1.set_title(f'Panel A: Q-Learning — Agent {agent} Price Evolution',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Greedy Price', fontsize=11)
    ax1.set_xlim(0, iters_ddpg[-1])
    ax1.set_ylim(1.1, 14.9)
    ax1.legend(loc='upper right')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax1.grid(True, alpha=0.3)

    # ── DDPG: mean and std of episode_prices across runs, smoothed ──
    all_ep_prices = np.array([run[f'agent{agent}']['episode_prices'] for run in runs_ddpg])
    mean_ep = all_ep_prices.mean(axis=0)
    std_ep  = all_ep_prices.std(axis=0)

    w_ddpg     = max(1, n_ep // 10)
    mean_ep_sm = smooth(mean_ep, w_ddpg)
    std_ep_sm  = smooth(std_ep,  w_ddpg)

    ax2.fill_between(iters_ddpg, mean_ep_sm - std_ep_sm, mean_ep_sm + std_ep_sm,
                     color=unibonn_grey, alpha=0.7, label='± 1 std.')
    ax2.plot(iters_ddpg, mean_ep_sm, color=unibonn_yellow, linewidth=2, label='Average')
    ax2.axhline(nash, color='black', linestyle='--', linewidth=1.5,
                label=f'Glosten-Milgrom Price ({nash:.2f})')
    ax2.axhline(monopoly_price, color=unibonn_grey, linestyle='--', linewidth=1.5,
                label=f'Monopoly Price ({monopoly_price:.2f})')
    ax2.set_title(f'Panel B: DDPG — Agent {agent} Price Evolution',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Price', fontsize=12)
    ax2.set_xlim(0, iters_ddpg[-1])
    ax2.set_ylim(1.1, 14.9)
    ax2.legend(loc='upper right')
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{THESIS_DIR}/compare_price_evo_agent{agent}_sigma{sigma}.pdf',
                bbox_inches='tight', dpi=300)
    plt.show()


def compare_qs_by_sigma():
    """
    Left:  plot_spreads_by_sigma  from qlearn_plot — Q-Learning QS vs sigma
    Right: plot_qs_rs_comparison  from DDPG_plot   — DDPG QS vs sigma
    Both panels show AS (blue) and NAS (yellow) with Nash benchmarks.
    """
    sigmas = np.array(SIGMAS)

    def ci(data):
        a  = np.array(data)
        se = np.std(a, ddof=1) / np.sqrt(len(a))
        m  = np.mean(a)
        return m, m - 1.96 * se, m + 1.96 * se

    # ── Collect Q-Learning data ──
    ql_as_means,  ql_as_lower,  ql_as_upper  = [], [], []
    ql_nas_means, ql_nas_lower, ql_nas_upper = [], [], []
    for s in sigmas:
        m, lo, hi = ci(load_ql(s)['quoted_spread'])
        ql_as_means.append(m); ql_as_lower.append(lo); ql_as_upper.append(hi)
        m, lo, hi = ci(load_ql_nas(s)['quoted_spread'])
        ql_nas_means.append(m); ql_nas_lower.append(lo); ql_nas_upper.append(hi)

    # ── Collect DDPG data (CI across per-run means, matching Q-learning) ──
    ddpg_as_means,  ddpg_as_lower,  ddpg_as_upper  = [], [], []
    ddpg_nas_means, ddpg_nas_lower, ddpg_nas_upper = [], [], []
    for s in sigmas:
        m, lo, hi = ci(_ddpg_run_means(s, 'quoted_spreads', nas=False))
        ddpg_as_means.append(m); ddpg_as_lower.append(lo); ddpg_as_upper.append(hi)
        m, lo, hi = ci(_ddpg_run_means(s, 'quoted_spreads', nas=True))
        ddpg_nas_means.append(m); ddpg_nas_lower.append(lo); ddpg_nas_upper.append(hi)

    # ── Nash benchmarks ──
    nash_as  = np.array([brentq(gm_equation, E_V, 8.0, args=(4, 0, 0.5, s)) - E_V
                         for s in sigmas])
    nash_nas = np.zeros(len(sigmas))

    # ── Shared axis limits ──
    all_hi = ql_as_upper + ql_nas_upper + ddpg_as_upper + ddpg_nas_upper + list(nash_as)
    y_min  = -0.1
    y_max  = max(all_hi) + 0.1
    x_min, x_max = sigmas[0] - 0.5, sigmas[-1] + 0.5

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, as_m, as_lo, as_hi, nas_m, nas_lo, nas_hi, title in [
        (ax1, ql_as_means,   ql_as_lower,   ql_as_upper,
              ql_nas_means,  ql_nas_lower,  ql_nas_upper,  'Panel A: Q-Learning'),
        (ax2, ddpg_as_means, ddpg_as_lower, ddpg_as_upper,
              ddpg_nas_means, ddpg_nas_lower, ddpg_nas_upper, 'Panel B: DDPG'),
    ]:
        ax.fill_between(sigmas, as_lo,  as_hi,  color=unibonn_blue,   alpha=0.30)
        ax.fill_between(sigmas, nas_lo, nas_hi, color=unibonn_yellow, alpha=0.30)
        ax.plot(sigmas, as_m,   color=unibonn_blue,   linewidth=2, marker='o', markersize=6, label='Adverse Selection')
        ax.plot(sigmas, nas_m,  color=unibonn_yellow, linewidth=2, marker='o', markersize=6, label='No Adverse Selection')
        ax.plot(sigmas, nash_as,  color=unibonn_blue,   linewidth=1.5, linestyle='--', label='Nash — Adverse Selection')
        ax.plot(sigmas, nash_nas, color=unibonn_yellow, linewidth=1.5, linestyle='--', label='Nash — No Adverse Selection')
        ax.set_title(title, fontsize=13, fontweight='bold', loc='center')
        ax.set_xlabel('$\\sigma$', fontsize=12)
        ax.set_ylabel('Mean Quoted Spread', fontsize=12)
        ax.set_xticks(sigmas)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(plt.AutoLocator())
        ax.legend(fontsize=9, loc='upper left', framealpha=0.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{THESIS_DIR}/compare_qs_by_sigma.pdf',
                bbox_inches='tight', dpi=300)
    plt.show()


def compare_rs_by_sigma():
    sigmas = np.array(SIGMAS)

    def ci(data):
        a  = np.array(data)
        se = np.std(a, ddof=1) / np.sqrt(len(a))
        m  = np.mean(a)
        return m, m - 1.96 * se, m + 1.96 * se

    # ── Collect Q-Learning data ──
    ql_as_means,  ql_as_lower,  ql_as_upper  = [], [], []
    ql_nas_means, ql_nas_lower, ql_nas_upper = [], [], []
    for s in sigmas:
        m, lo, hi = ci(load_ql(s)['realized_spread'])
        ql_as_means.append(m); ql_as_lower.append(lo); ql_as_upper.append(hi)
        m, lo, hi = ci(load_ql_nas(s)['realized_spread'])
        ql_nas_means.append(m); ql_nas_lower.append(lo); ql_nas_upper.append(hi)

    # ── Collect DDPG data (CI across per-run means, matching Q-learning) ──
    ddpg_as_means,  ddpg_as_lower,  ddpg_as_upper  = [], [], []
    ddpg_nas_means, ddpg_nas_lower, ddpg_nas_upper = [], [], []
    for s in sigmas:
        m, lo, hi = ci(_ddpg_run_means(s, 'realized_spreads', nas=False))
        ddpg_as_means.append(m); ddpg_as_lower.append(lo); ddpg_as_upper.append(hi)
        m, lo, hi = ci(_ddpg_run_means(s, 'realized_spreads', nas=True))
        ddpg_nas_means.append(m); ddpg_nas_lower.append(lo); ddpg_nas_upper.append(hi)

    # ── Nash benchmarks (RS = 0 for both AS and NAS at equilibrium) ──
    nash_as  = np.zeros(len(sigmas))
    nash_nas = np.zeros(len(sigmas))

    # ── Shared axis limits ──
    all_hi = ql_as_upper + ql_nas_upper + ddpg_as_upper + ddpg_nas_upper
    y_min  = -0.1
    y_max  = max(all_hi) + 0.1
    x_min, x_max = sigmas[0] - 0.5, sigmas[-1] + 0.5

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, as_m, as_lo, as_hi, nas_m, nas_lo, nas_hi, title in [
        (ax1, ql_as_means,   ql_as_lower,   ql_as_upper,
              ql_nas_means,  ql_nas_lower,  ql_nas_upper,  'Panel A: Q-Learning'),
        (ax2, ddpg_as_means, ddpg_as_lower, ddpg_as_upper,
              ddpg_nas_means, ddpg_nas_lower, ddpg_nas_upper, 'Panel B: DDPG'),
    ]:
        ax.fill_between(sigmas, as_lo,  as_hi,  color=unibonn_blue,   alpha=0.30)
        ax.fill_between(sigmas, nas_lo, nas_hi, color=unibonn_yellow, alpha=0.30)
        ax.plot(sigmas, as_m,   color=unibonn_blue,   linewidth=2, marker='o', markersize=6, label='Adverse Selection')
        ax.plot(sigmas, nas_m,  color=unibonn_yellow, linewidth=2, marker='o', markersize=6, label='No Adverse Selection')
        ax.plot(sigmas, nash_as,  color=unibonn_blue,   linewidth=1.5, linestyle='--', label='Nash — Adverse Selection')
        ax.plot(sigmas, nash_nas, color=unibonn_yellow, linewidth=1.5, linestyle='--', label='Nash — No Adverse Selection')
        ax.set_title(title, fontsize=13, fontweight='bold', loc='center')
        ax.set_xlabel('$\\sigma$', fontsize=12)
        ax.set_ylabel('Mean Realized Spread', fontsize=12)
        ax.set_xticks(sigmas)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(plt.AutoLocator())
        ax.legend(fontsize=9, loc='upper left', framealpha=0.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{THESIS_DIR}/compare_rs_by_sigma.pdf',
                bbox_inches='tight', dpi=300)
    plt.show()


def compare_qs_by_N(sigma=SIGMA):
    """
    Quoted spread vs number of agents N ∈ {2,3,4,5} at fixed sigma.
    Panel A: Q-Learning  |  Panel B: DDPG
    AS in unibonn_blue, NAS in unibonn_yellow, 95% CI bands, Nash benchmarks.
    """
    ns = np.array([2, 3, 4, 5])

    def ci(data):
        a  = np.array(data)
        se = np.std(a, ddof=1) / np.sqrt(len(a))
        m  = np.mean(a)
        return m, m - 1.96 * se, m + 1.96 * se

    ql_as_means,  ql_as_lower,  ql_as_upper  = [], [], []
    ql_nas_means, ql_nas_lower, ql_nas_upper = [], [], []
    for n in ns:
        m, lo, hi = ci(_ql_spread_N(n, 'quoted_spread', sigma=sigma, nas=False))
        ql_as_means.append(m); ql_as_lower.append(lo); ql_as_upper.append(hi)
        m, lo, hi = ci(_ql_spread_N(n, 'quoted_spread', sigma=sigma, nas=True))
        ql_nas_means.append(m); ql_nas_lower.append(lo); ql_nas_upper.append(hi)

    ddpg_as_means,  ddpg_as_lower,  ddpg_as_upper  = [], [], []
    ddpg_nas_means, ddpg_nas_lower, ddpg_nas_upper = [], [], []
    for n in ns:
        m, lo, hi = ci(_ddpg_run_means_N(n, 'quoted_spreads', sigma=sigma, nas=False))
        ddpg_as_means.append(m); ddpg_as_lower.append(lo); ddpg_as_upper.append(hi)
        m, lo, hi = ci(_ddpg_run_means_N(n, 'quoted_spreads', sigma=sigma, nas=True))
        ddpg_nas_means.append(m); ddpg_nas_lower.append(lo); ddpg_nas_upper.append(hi)

    # Nash benchmark: same GM price regardless of N (Bertrand zero-profit condition)
    nash_qs_as  = brentq(gm_equation, E_V, 8.0, args=(4, 0, 0.5, sigma)) - E_V
    nash_qs_nas = 0.0

    all_hi = ql_as_upper + ql_nas_upper + ddpg_as_upper + ddpg_nas_upper
    y_min  = -0.1
    y_max  = max(all_hi + [nash_qs_as]) + 0.1
    x_min, x_max = ns[0] - 0.3, ns[-1] + 0.3

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, as_m, as_lo, as_hi, nas_m, nas_lo, nas_hi, title in [
        (ax1, ql_as_means,   ql_as_lower,   ql_as_upper,
              ql_nas_means,  ql_nas_lower,  ql_nas_upper,  'Panel A: Q-Learning'),
        (ax2, ddpg_as_means, ddpg_as_lower, ddpg_as_upper,
              ddpg_nas_means, ddpg_nas_lower, ddpg_nas_upper, 'Panel B: DDPG'),
    ]:
        ax.fill_between(ns, as_lo,  as_hi,  color=unibonn_blue,   alpha=0.30)
        ax.fill_between(ns, nas_lo, nas_hi, color=unibonn_yellow, alpha=0.30)
        ax.plot(ns, as_m,  color=unibonn_blue,   linewidth=2, marker='o', markersize=6, label='Adverse Selection')
        ax.plot(ns, nas_m, color=unibonn_yellow, linewidth=2, marker='o', markersize=6, label='No Adverse Selection')
        ax.axhline(nash_qs_as,  color=unibonn_blue,   linewidth=1.5, linestyle='--', label='Nash — Adverse Selection')
        ax.axhline(nash_qs_nas, color=unibonn_yellow, linewidth=1.5, linestyle='--', label='Nash — No Adverse Selection')
        ax.set_title(title, fontsize=13, fontweight='bold', loc='center')
        ax.set_xlabel('$N$ (number of agents)', fontsize=12)
        ax.set_ylabel('Mean Quoted Spread', fontsize=12)
        ax.set_xticks(ns)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(plt.AutoLocator())
        ax.legend(fontsize=9, loc='upper right', framealpha=0.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{THESIS_DIR}/compare_qs_by_N_sigma{sigma}.pdf',
                bbox_inches='tight', dpi=300)
    plt.show()


def compare_rs_by_N(sigma=SIGMA):
    """
    Realized spread vs number of agents N ∈ {2,3,4,5} at fixed sigma.
    Panel A: Q-Learning  |  Panel B: DDPG
    AS in unibonn_blue, NAS in unibonn_yellow, 95% CI bands, Nash benchmarks.
    """
    ns = np.array([2, 3, 4, 5])

    def ci(data):
        a  = np.array(data)
        se = np.std(a, ddof=1) / np.sqrt(len(a))
        m  = np.mean(a)
        return m, m - 1.96 * se, m + 1.96 * se

    ql_as_means,  ql_as_lower,  ql_as_upper  = [], [], []
    ql_nas_means, ql_nas_lower, ql_nas_upper = [], [], []
    for n in ns:
        m, lo, hi = ci(_ql_spread_N(n, 'realized_spread', sigma=sigma, nas=False))
        ql_as_means.append(m); ql_as_lower.append(lo); ql_as_upper.append(hi)
        m, lo, hi = ci(_ql_spread_N(n, 'realized_spread', sigma=sigma, nas=True))
        ql_nas_means.append(m); ql_nas_lower.append(lo); ql_nas_upper.append(hi)

    ddpg_as_means,  ddpg_as_lower,  ddpg_as_upper  = [], [], []
    ddpg_nas_means, ddpg_nas_lower, ddpg_nas_upper = [], [], []
    for n in ns:
        m, lo, hi = ci(_ddpg_run_means_N(n, 'realized_spreads', sigma=sigma, nas=False))
        ddpg_as_means.append(m); ddpg_as_lower.append(lo); ddpg_as_upper.append(hi)
        m, lo, hi = ci(_ddpg_run_means_N(n, 'realized_spreads', sigma=sigma, nas=True))
        ddpg_nas_means.append(m); ddpg_nas_lower.append(lo); ddpg_nas_upper.append(hi)

    nash_rs = 0.0  # Nash RS = 0 for both AS and NAS at equilibrium

    all_hi = ql_as_upper + ql_nas_upper + ddpg_as_upper + ddpg_nas_upper
    y_min  = -0.1
    y_max  = max(all_hi) + 0.1
    x_min, x_max = ns[0] - 0.3, ns[-1] + 0.3

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, as_m, as_lo, as_hi, nas_m, nas_lo, nas_hi, title in [
        (ax1, ql_as_means,   ql_as_lower,   ql_as_upper,
              ql_nas_means,  ql_nas_lower,  ql_nas_upper,  'Panel A: Q-Learning'),
        (ax2, ddpg_as_means, ddpg_as_lower, ddpg_as_upper,
              ddpg_nas_means, ddpg_nas_lower, ddpg_nas_upper, 'Panel B: DDPG'),
    ]:
        ax.fill_between(ns, as_lo,  as_hi,  color=unibonn_blue,   alpha=0.30)
        ax.fill_between(ns, nas_lo, nas_hi, color=unibonn_yellow, alpha=0.30)
        ax.plot(ns, as_m,  color=unibonn_blue,   linewidth=2, marker='o', markersize=6, label='Adverse Selection')
        ax.plot(ns, nas_m, color=unibonn_yellow, linewidth=2, marker='o', markersize=6, label='No Adverse Selection')
        ax.axhline(nash_rs, color=unibonn_grey, linewidth=1.5, linestyle='--', label='Nash (both)')
        ax.set_title(title, fontsize=13, fontweight='bold', loc='center')
        ax.set_xlabel('$N$ (number of agents)', fontsize=12)
        ax.set_ylabel('Mean Realized Spread', fontsize=12)
        ax.set_xticks(ns)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(plt.AutoLocator())
        ax.legend(fontsize=9, loc='upper right', framealpha=0.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{THESIS_DIR}/compare_rs_by_N_sigma{sigma}.pdf',
                bbox_inches='tight', dpi=300)
    plt.show()


def comparison_table():
    """
    LaTeX table for sigma=5: Agent 1 and Agent 2 prices (exploitation phase).
    Columns: GM Benchmark | Colliard et al. (2025) [blank] | Q-Learning | DDPG
    Rows: mean, std, t-stat, p-value for each price.
    Unit of observation: per-experiment exploitation-phase mean (10 per algorithm).
    """
    sigma = 5

    data_ddpg = np.load(f"results_sigma{sigma}_N{N}.npy", allow_pickle=True)
    runs_as   = list(data_ddpg) if data_ddpg.ndim != 0 else [data_ddpg.item()]
    ql = load_ql(sigma)

    nash_price = brentq(gm_equation, E_V, 8.0, args=(4, 0, 0.5, sigma))

    # One mean per experiment: average exploitation-phase price across all steps in that run
    ddpg_p1 = np.array([np.mean([r['price_1'] for r in run['exploitation']]) for run in runs_as])
    ddpg_p2 = np.array([np.mean([r['price_2'] for r in run['exploitation']]) for run in runs_as])
    ql_p1   = np.array([np.mean(prices) for prices in ql['exploit_price_1']])
    ql_p2   = np.array([np.mean(prices) for prices in ql['exploit_price_2']])

    # One-sample t-test: H0 — exploitation-phase mean equals Nash benchmark
    ql_p1_t,   ql_p1_p   = ttest_1samp(ql_p1,   nash_price)
    ql_p2_t,   ql_p2_p   = ttest_1samp(ql_p2,   nash_price)
    ddpg_p1_t, ddpg_p1_p = ttest_1samp(ddpg_p1, nash_price)
    ddpg_p2_t, ddpg_p2_p = ttest_1samp(ddpg_p2, nash_price)

    def fmt(arr):
        return f"{np.mean(arr):.2f}", f"{np.std(arr, ddof=1):.2f}"

    def stars(p):
        if p < 0.01: return r"$^{***}$"
        if p < 0.05: return r"$^{**}$"
        if p < 0.10: return r"$^{*}$"
        return ""

    ql_p1_m,   ql_p1_s   = fmt(ql_p1)
    ql_p2_m,   ql_p2_s   = fmt(ql_p2)
    ddpg_p1_m, ddpg_p1_s = fmt(ddpg_p1)
    ddpg_p2_m, ddpg_p2_s = fmt(ddpg_p2)

    tex = (
        r"\begin{tabular}{lcccc}" + "\n"
        r"\hline" + "\n"
        r"& \textbf{GM Benchmark} & \textbf{Colliard et al.\ (2025)} "
        r"& \textbf{Q-Learning} & \textbf{DDPG} \\" + "\n"
        r"\hline" + "\n"
        r"\multicolumn{5}{l}{\textit{Panel A: Agent 1 Price}} \\" + "\n"
        f"Mean      & {nash_price:.2f} & 4.97 & {ql_p1_m} & {ddpg_p1_m} \\\\\n"
        f"Std       & ---             & 0.73 & {ql_p1_s} & {ddpg_p1_s} \\\\\n"
        f"$t$-stat  & ---             & ---  & {ql_p1_t:.2f}{stars(ql_p1_p)} & {ddpg_p1_t:.2f}{stars(ddpg_p1_p)} \\\\\n"
        r"\hline" + "\n"
        r"\multicolumn{5}{l}{\textit{Panel B: Agent 2 Price}} \\" + "\n"
        f"Mean      & {nash_price:.2f} & 4.97 & {ql_p2_m} & {ddpg_p2_m} \\\\\n"
        f"Std       & ---             & 0.73 & {ql_p2_s} & {ddpg_p2_s} \\\\\n"
        f"$t$-stat  & ---             & ---  & {ql_p2_t:.2f}{stars(ql_p2_p)} & {ddpg_p2_t:.2f}{stars(ddpg_p2_p)} \\\\\n"
        r"\hline" + "\n"
        r"\end{tabular}" + "\n"
    )

    print(tex)
    path = f'{THESIS_DIR}/comparison_table.tex'
    with open(path, 'w') as f:
        f.write(tex)
    print(f"Saved to {path}")


def spread_table():
    """
    LaTeX table for sigma=5: Quoted Spread and Realized Spread (exploitation phase).
    Columns: GM Benchmark | Colliard et al. (2025) [blank] | Q-Learning | DDPG
    Rows: mean and std for each spread.
    """
    sigma = 5

    data_ddpg = np.load(f"results_sigma{sigma}_N{N}.npy", allow_pickle=True)
    runs_as   = list(data_ddpg) if data_ddpg.ndim != 0 else [data_ddpg.item()]
    ql = load_ql(sigma)

    nash_price = brentq(gm_equation, E_V, 8.0, args=(4, 0, 0.5, sigma))
    nash_qs    = nash_price - E_V
    nash_rs    = 0.0

    # DDPG: step-level spreads pooled across all runs
    ddpg_qs_arr = np.array([s for run in runs_as for s in run['quoted_spreads']])
    ddpg_rs_arr = np.array([s for run in runs_as for s in run['realized_spreads']])

    # Q-learning: per-experiment mean spreads (100 values each)
    ql_qs_arr = np.array(ql['quoted_spread'])
    ql_rs_arr = np.array(ql['realized_spread'])

    def fmt(arr):
        return f"{np.mean(arr):.3f}", f"{np.std(arr, ddof=1):.3f}"

    ql_qs_m,   ql_qs_s   = fmt(ql_qs_arr)
    ql_rs_m,   ql_rs_s   = fmt(ql_rs_arr)
    ddpg_qs_m, ddpg_qs_s = fmt(ddpg_qs_arr)
    ddpg_rs_m, ddpg_rs_s = fmt(ddpg_rs_arr)

    tex = (
        r"\begin{table}[H]" + "\n"
        r"\centering" + "\n"
        r"\caption{Exploitation-Phase Spreads for $\sigma=5$, $N=2$}" + "\n"
        r"\label{tab:spreads}" + "\n"
        r"\begin{tabular}{llcccc}" + "\n"
        r"\hline" + "\n"
        r"\textbf{Metric} & & \textbf{GM Benchmark} & \textbf{Colliard et al.\ (2025)} "
        r"& \textbf{Q-Learning} & \textbf{DDPG} \\" + "\n"
        r"\hline" + "\n"
        r"\multirow{2}{*}{Quoted Spread} "
        f"& Mean & {nash_qs:.3f} &  & {ql_qs_m} & {ddpg_qs_m} \\\\\n"
        f"& Std  & ---          &  & {ql_qs_s} & {ddpg_qs_s} \\\\\n"
        r"\hline" + "\n"
        r"\multirow{2}{*}{Realized Spread} "
        f"& Mean & {nash_rs:.3f} &  & {ql_rs_m} & {ddpg_rs_m} \\\\\n"
        f"& Std  & ---          &  & {ql_rs_s} & {ddpg_rs_s} \\\\\n"
        r"\hline" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}" + "\n"
    )

    print(tex)
    path = f'{THESIS_DIR}/spread_table.tex'
    with open(path, 'w') as f:
        f.write(tex)
    print(f"Saved to {path}")


def summary_sigma5():
    """
    Prints mean / std / median for agent 1 price, agent 2 price, quoted spread,
    and realized spread for sigma=5, across all four variants:
    Q-Learning AS, Q-Learning NAS, DDPG AS, DDPG NAS.
    Also shows the Nash benchmark price and spreads.
    """
    sigma = 5

    # ── Load data ──
    data_ddpg     = np.load(f"results_sigma{sigma}_N{N}.npy",     allow_pickle=True)
    data_ddpg_nas = np.load(f"results_nas_sigma{sigma}_N{N}.npy", allow_pickle=True)
    runs_as  = list(data_ddpg)     if data_ddpg.ndim     != 0 else [data_ddpg.item()]
    runs_nas = list(data_ddpg_nas) if data_ddpg_nas.ndim != 0 else [data_ddpg_nas.item()]
    ql     = load_ql(sigma)
    ql_nas = load_ql_nas(sigma)

    # ── Nash benchmark ──
    nash_price = brentq(gm_equation, E_V, 8.0, args=(4, 0, 0.5, sigma))
    nash_qs    = nash_price - E_V
    nash_rs    = 0.0

    def stats(arr):
        a = np.asarray(arr, dtype=float)
        return np.mean(a), np.std(a, ddof=1), np.median(a)

    # ── Extract arrays ──
    # Q-Learning: exploitation-phase prices pooled across all experiments;
    # spreads are per-experiment means
    ql_p1  = np.concatenate(ql['exploit_price_1'])
    ql_p2  = np.concatenate(ql['exploit_price_2'])
    ql_qs  = np.array(ql['quoted_spread'])
    ql_rs  = np.array(ql['realized_spread'])

    ql_nas_p1 = np.concatenate(ql_nas['exploit_price_1'])
    ql_nas_p2 = np.concatenate(ql_nas['exploit_price_2'])
    ql_nas_qs = np.array(ql_nas['quoted_spread'])
    ql_nas_rs = np.array(ql_nas['realized_spread'])

    # DDPG: exploitation step-level prices and spreads, pooled across all runs
    ddpg_p1 = np.array([r['price_1'] for run in runs_as  for r in run['exploitation']])
    ddpg_p2 = np.array([r['price_2'] for run in runs_as  for r in run['exploitation']])
    ddpg_qs = np.array([s for run in runs_as  for s in run['quoted_spreads']])
    ddpg_rs = np.array([s for run in runs_as  for s in run['realized_spreads']])

    ddpg_nas_p1 = np.array([r['price_1'] for run in runs_nas for r in run['exploitation']])
    ddpg_nas_p2 = np.array([r['price_2'] for run in runs_nas for r in run['exploitation']])
    ddpg_nas_qs = np.array([s for run in runs_nas for s in run['quoted_spreads']])
    ddpg_nas_rs = np.array([s for run in runs_nas for s in run['realized_spreads']])

    # ── Print summary ──
    col_w = 12
    header = (f"{'Metric':<22} {'Nash':>{col_w}} "
              f"{'QL-AS':>{col_w}} {'QL-NAS':>{col_w}} "
              f"{'DDPG-AS':>{col_w}} {'DDPG-NAS':>{col_w}}")
    sep = '-' * len(header)

    def row(label, nash_val, ql_v, ql_nas_v, ddpg_v, ddpg_nas_v, stat_fn):
        m1,  s1,  md1  = stat_fn(ql_v)
        m2,  s2,  md2  = stat_fn(ql_nas_v)
        m3,  s3,  md3  = stat_fn(ddpg_v)
        m4,  s4,  md4  = stat_fn(ddpg_nas_v)
        nv = f'{nash_val:.3f}' if nash_val is not None else '—'
        lines = [
            f"  {'mean':<20} {nv:>{col_w}} {m1:>{col_w}.3f} {m2:>{col_w}.3f} {m3:>{col_w}.3f} {m4:>{col_w}.3f}",
            f"  {'std':<20} {'—':>{col_w}} {s1:>{col_w}.3f} {s2:>{col_w}.3f} {s3:>{col_w}.3f} {s4:>{col_w}.3f}",
            f"  {'median':<20} {'—':>{col_w}} {md1:>{col_w}.3f} {md2:>{col_w}.3f} {md3:>{col_w}.3f} {md4:>{col_w}.3f}",
        ]
        return label + '\n' + '\n'.join(lines)

    print(f"\nSummary — sigma={sigma}, N={N}")
    print(sep)
    print(header)
    print(sep)
    print(row('Agent 1 Price',  nash_price, ql_p1,  ql_nas_p1, ddpg_p1,  ddpg_nas_p1, stats))
    print(sep)
    print(row('Agent 2 Price',  nash_price, ql_p2,  ql_nas_p2, ddpg_p2,  ddpg_nas_p2, stats))
    print(sep)
    print(row('Quoted Spread',  nash_qs,    ql_qs,  ql_nas_qs, ddpg_qs,  ddpg_nas_qs, stats))
    print(sep)
    print(row('Realized Spread', nash_rs,   ql_rs,  ql_nas_rs, ddpg_rs,  ddpg_nas_rs, stats))
    print(sep)


def summary_training_prices_sigma5():
    """
    Prints mean / std / median for agent 1 and agent 2 prices from the
    training phase only, for sigma=5, across all four variants.
    Q-Learning: final converged greedy price per experiment (final_prices_1/2).
    DDPG: greedy (no-noise) episode price snapshots pooled across all runs.
    """
    sigma = 5

    data_ddpg     = np.load(f"results_sigma{sigma}_N{N}.npy",     allow_pickle=True)
    data_ddpg_nas = np.load(f"results_nas_sigma{sigma}_N{N}.npy", allow_pickle=True)
    runs_as  = list(data_ddpg)     if data_ddpg.ndim     != 0 else [data_ddpg.item()]
    runs_nas = list(data_ddpg_nas) if data_ddpg_nas.ndim != 0 else [data_ddpg_nas.item()]
    ql     = load_ql(sigma)
    ql_nas = load_ql_nas(sigma)

    nash_price = brentq(gm_equation, E_V, 8.0, args=(4, 0, 0.5, sigma))

    def stats(arr):
        a = np.asarray(arr, dtype=float)
        return np.mean(a), np.std(a, ddof=1), np.median(a)

    # Q-Learning: one final greedy price per experiment
    ql_p1     = np.array(ql['final_prices_1'])
    ql_p2     = np.array(ql['final_prices_2'])
    ql_nas_p1 = np.array(ql_nas['final_prices_1'])
    ql_nas_p2 = np.array(ql_nas['final_prices_2'])

    # DDPG: all greedy episode price snapshots pooled across runs
    ddpg_p1     = np.concatenate([run['agent1']['episode_prices'] for run in runs_as])
    ddpg_p2     = np.concatenate([run['agent2']['episode_prices'] for run in runs_as])
    ddpg_nas_p1 = np.concatenate([run['agent1']['episode_prices'] for run in runs_nas])
    ddpg_nas_p2 = np.concatenate([run['agent2']['episode_prices'] for run in runs_nas])

    col_w = 12
    header = (f"{'Metric':<22} {'Nash':>{col_w}} "
              f"{'QL-AS':>{col_w}} {'QL-NAS':>{col_w}} "
              f"{'DDPG-AS':>{col_w}} {'DDPG-NAS':>{col_w}}")
    sep = '-' * len(header)

    def row(label, nash_val, v1, v2, v3, v4):
        m1, s1, md1 = stats(v1)
        m2, s2, md2 = stats(v2)
        m3, s3, md3 = stats(v3)
        m4, s4, md4 = stats(v4)
        nv = f'{nash_val:.3f}'
        lines = [
            f"  {'mean':<20} {nv:>{col_w}} {m1:>{col_w}.3f} {m2:>{col_w}.3f} {m3:>{col_w}.3f} {m4:>{col_w}.3f}",
            f"  {'std':<20} {'—':>{col_w}} {s1:>{col_w}.3f} {s2:>{col_w}.3f} {s3:>{col_w}.3f} {s4:>{col_w}.3f}",
            f"  {'median':<20} {'—':>{col_w}} {md1:>{col_w}.3f} {md2:>{col_w}.3f} {md3:>{col_w}.3f} {md4:>{col_w}.3f}",
        ]
        return label + '\n' + '\n'.join(lines)

    print(f"\nTraining-Phase Price Summary — sigma={sigma}, N={N}")
    print(sep)
    print(header)
    print(sep)
    print(row('Agent 1 Price', nash_price, ql_p1, ql_nas_p1, ddpg_p1, ddpg_nas_p1))
    print(sep)
    print(row('Agent 2 Price', nash_price, ql_p2, ql_nas_p2, ddpg_p2, ddpg_nas_p2))
    print(sep)


if __name__ == "__main__":
    plot_price_distribution()
    plot_price_distribution(agent=2)
    compare_price_evolution()
    compare_price_evolution(agent=2)
    compare_qs_by_sigma()
    compare_rs_by_sigma()
    compare_qs_by_N()
    compare_rs_by_N()
    comparison_table()
    spread_table()
    summary_sigma5()
    summary_training_prices_sigma5()
