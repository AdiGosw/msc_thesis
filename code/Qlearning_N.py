# %%
import numpy as np
import random
from typing import List, Tuple, Dict
from scipy import stats as st
import time
from joblib import Parallel, delayed

# Global Seed
SEED = 67
np.random.seed(SEED)
random.seed(SEED)


# ── Environments ──────────────────────────────────────────────────────────────

class EnvironmentAS:
    """Adverse selection: client valuation = realised asset value + liquidity shock."""
    def __init__(self, v_high=4.0, v_low=0.0, mu=0.5, sigma=5.0):
        self.v_high = v_high
        self.v_low  = v_low
        self.mu     = mu
        self.sigma  = sigma
        self.current_asset_value = None
        self.reset()

    def reset(self):
        self.current_asset_value = (self.v_high if random.random() < self.mu
                                    else self.v_low)
        return self.current_asset_value

    def step(self, prices: List[float]) -> Tuple[List[float], bool, dict]:
        min_price       = min(prices)
        num_best_offers = prices.count(min_price)

        liquidity_shock  = np.random.normal(0, self.sigma)
        client_valuation = self.current_asset_value + liquidity_shock
        trade_occurred   = client_valuation >= min_price

        profits = []
        for price in prices:
            if trade_occurred and price == min_price:
                profits.append((min_price - self.current_asset_value) / num_best_offers)
            else:
                profits.append(0.0)

        return profits, trade_occurred, {
            'asset_value':     self.current_asset_value,
            'client_valuation': client_valuation,
            'min_price':       min_price,
            'trade_occurred':  trade_occurred,
        }


class EnvironmentNAS:
    """No adverse selection: client draws their own independent value."""
    def __init__(self, v_high=4.0, v_low=0.0, mu=0.5, sigma=5.0):
        self.v_high = v_high
        self.v_low  = v_low
        self.mu     = mu
        self.sigma  = sigma
        self.current_asset_value = None
        self.reset()

    def reset(self):
        self.current_asset_value = (self.v_high if random.random() < self.mu
                                    else self.v_low)
        return self.current_asset_value

    def step(self, prices: List[float]) -> Tuple[List[float], bool, dict]:
        min_price       = min(prices)
        num_best_offers = prices.count(min_price)

        liquidity_shock  = np.random.normal(0, self.sigma)
        client_value     = (self.v_high if random.random() < self.mu else self.v_low)
        client_valuation = client_value + liquidity_shock
        trade_occurred   = client_valuation >= min_price

        profits = []
        for price in prices:
            if trade_occurred and price == min_price:
                profits.append((min_price - self.current_asset_value) / num_best_offers)
            else:
                profits.append(0.0)

        return profits, trade_occurred, {
            'asset_value':     self.current_asset_value,
            'client_valuation': client_valuation,
            'min_price':       min_price,
            'trade_occurred':  trade_occurred,
        }


# ── Q-Agent ───────────────────────────────────────────────────────────────────

class QAgent:
    def __init__(self, price_grid: List[float], alpha: float = 0.01,
                 beta: float = 8.1e-5, initial_q: Tuple[float, float] = (3.0, 6.0),
                 stable_episodes_required: int = 50000):
        self.price_grid   = price_grid
        self.num_prices   = len(price_grid)
        self.alpha        = alpha
        self.beta         = beta
        self.stable_episodes_required = stable_episodes_required

        self.q_values = np.random.uniform(initial_q[0], initial_q[1], self.num_prices)
        self.episode  = 0

        self.converged             = False
        self.convergence_episode   = None
        self.stable_action_episodes = 0
        self.last_greedy_action    = None

        self.final_q_values      = None
        self.final_greedy_action = None
        self.learning_complete   = False

    def get_epsilon(self) -> float:
        return np.exp(-self.beta * self.episode)

    def choose_action_learning(self) -> int:
        if random.random() < self.get_epsilon():
            return random.randint(0, self.num_prices - 1)
        return int(np.argmax(self.q_values))

    def choose_action_exploitation(self) -> int:
        return self.final_greedy_action

    def update_q_value(self, price_index: int, profit: float):
        self.q_values[price_index] = (self.alpha * profit +
                                      (1 - self.alpha) * self.q_values[price_index])

    def check_convergence(self):
        if self.converged:
            return True
        current = int(np.argmax(self.q_values))
        if self.last_greedy_action is None:
            self.last_greedy_action     = current
            self.stable_action_episodes = 1
        elif current == self.last_greedy_action:
            self.stable_action_episodes += 1
        else:
            self.stable_action_episodes = 1
            self.last_greedy_action     = current
        if self.stable_action_episodes >= self.stable_episodes_required:
            self.converged           = True
            self.convergence_episode = self.episode
            self.final_q_values      = self.q_values.copy()
            self.final_greedy_action = current
        return self.converged

    def finalize_learning(self):
        if not self.converged:
            self.final_q_values      = self.q_values.copy()
            self.final_greedy_action = int(np.argmax(self.q_values))
        self.learning_complete = True

    def step_learning(self) -> Tuple[int, float]:
        action = self.choose_action_learning()
        return action, self.price_grid[action]

    def step_exploitation(self) -> Tuple[int, float]:
        action = self.choose_action_exploitation()
        return action, self.price_grid[action]

    def learn(self, action: int, profit: float):
        self.update_q_value(action, profit)
        self.episode += 1
        self.check_convergence()

    def get_greedy_price(self) -> float:
        if self.learning_complete:
            return self.price_grid[self.final_greedy_action]
        return self.price_grid[int(np.argmax(self.q_values))]

    def is_converged(self) -> bool:
        return self.converged

    def is_learning_complete(self) -> bool:
        return self.learning_complete


# ── Single experiment (generalised to N agents) ────────────────────────────────

def run_single_experiment(experiment_id, N=2, nas=False, track_every=1, sigma=5.0):
    """Run one experiment with N Q-learning agents."""
    env_class = EnvironmentNAS if nas else EnvironmentAS
    env       = env_class(v_high=4.0, v_low=0.0, mu=0.5, sigma=sigma)
    price_grid = [round(p, 1) for p in np.arange(1.1, 14.9, 0.1)]

    agents = [QAgent(price_grid, stable_episodes_required=50000) for _ in range(N)]

    price_history = []
    T = 500000

    # ── Learning Phase ──
    for step in range(T):
        env.reset()
        step_actions = [agent.step_learning() for agent in agents]
        action_ids   = [a[0] for a in step_actions]
        prices       = [a[1] for a in step_actions]

        profits, trade_occurred, info = env.step(prices)

        for i, agent in enumerate(agents):
            agent.learn(action_ids[i], profits[i])

        if step % track_every == 0:
            greedy_prices = [agent.get_greedy_price() for agent in agents]
            price_history.append({
                'step':            step,
                'greedy_price_1':  greedy_prices[0],
                'min_greedy_price': min(greedy_prices),
            })

        if all(agent.is_converged() for agent in agents):
            learning_steps = step + 1
            break
    else:
        learning_steps = T

    for agent in agents:
        agent.finalize_learning()

    final_greedy_prices = [agent.get_greedy_price() for agent in agents]
    final_min_price     = min(final_greedy_prices)
    expected_asset_value = 2.0

    # ── Exploitation Phase ──
    exploitation_results = []
    exploit_prices       = [[] for _ in range(N)]

    for _ in range(10000):
        asset_value  = env.reset()
        step_actions = [agent.step_exploitation() for agent in agents]
        prices       = [a[1] for a in step_actions]

        profits, trade_occurred, info = env.step(prices)

        for i in range(N):
            exploit_prices[i].append(prices[i])

        exploitation_results.append({
            'prices':         prices,
            'profits':        profits,
            'min_price':      min(prices),
            'asset_value':    asset_value,
            'trade_occurred': trade_occurred,
        })

    quoted_spreads   = [r['min_price'] - expected_asset_value
                        for r in exploitation_results]
    realized_spreads = [r['min_price'] - r['asset_value']
                        for r in exploitation_results if r['trade_occurred']]

    result = {
        'experiment_id': experiment_id,
        'learning_steps': learning_steps,
        'converged':      all(agent.is_converged() for agent in agents),
        'final_min_price': final_min_price,
        'quoted_spread':  np.mean(quoted_spreads),
        'realized_spread': np.mean(realized_spreads) if realized_spreads else 0.0,
        'price_history':  price_history,
    }
    for i in range(N):
        result[f'final_price_{i+1}']        = final_greedy_prices[i]
        result[f'exploitation_prices_{i+1}'] = exploit_prices[i]
        result[f'avg_profit_{i+1}']         = np.mean([r['profits'][i]
                                                        for r in exploitation_results])
    return result


# ── Full simulation for one (N, nas) condition ─────────────────────────────────

def run_full_simulation(N=3, nas=False, K=10, n_jobs=10, sigma=5.0):
    label = f"N={N} ({'NAS' if nas else 'AS'})"
    print(f"Running {K} experiments — {label} (sigma={sigma})...")
    start_time = time.time()

    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(run_single_experiment)(i, N=N, nas=nas, track_every=100, sigma=sigma)
        for i in range(K)
    )

    total_time = time.time() - start_time
    print(f"{label} completed in {total_time:.1f}s ({total_time/60:.1f} min)")

    save_dict = {
        'quoted_spread':  [r['quoted_spread']  for r in results],
        'realized_spread': [r['realized_spread'] for r in results],
        'final_min_price': [r['final_min_price'] for r in results],
        'learning_steps':  [r['learning_steps']  for r in results],
        'converged':       [r['converged']        for r in results],
        'price_history':   [r['price_history']    for r in results],
        'sigma': sigma,
        'N':     N,
    }
    for i in range(1, N + 1):
        save_dict[f'final_prices_{i}']  = [r[f'final_price_{i}']        for r in results]
        save_dict[f'exploit_price_{i}'] = [r[f'exploitation_prices_{i}'] for r in results]
        save_dict[f'avg_profit_{i}']    = [r[f'avg_profit_{i}']          for r in results]

    suffix   = '_nas' if nas else ''
    filename = f'qlearning_sigma{int(sigma)}_N{N}{suffix}.npy'
    np.save(filename, save_dict)
    print(f"Saved to {filename}")

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SIGMA  = 5
    NS     = [3, 4, 5]   # N=2 already exists from Qlearning.py / Qlearning_nas.py
    K      = 10
    N_JOBS = 10

    total_start = time.time()

    # ── Adverse Selection ──
    print(f"\n{'='*60}")
    print(f"ADVERSE SELECTION — sigma={SIGMA}, N in {NS}")
    print(f"{'='*60}")
    for n in NS:
        run_full_simulation(N=n, nas=False, K=K, n_jobs=N_JOBS, sigma=SIGMA)

    # ── No Adverse Selection ──
    print(f"\n{'='*60}")
    print(f"NO ADVERSE SELECTION — sigma={SIGMA}, N in {NS}")
    print(f"{'='*60}")
    for n in NS:
        run_full_simulation(N=n, nas=True, K=K, n_jobs=N_JOBS, sigma=SIGMA)

    total_time = time.time() - total_start
    print(f"\nAll done in {total_time:.1f}s ({total_time/60:.1f} min)")
