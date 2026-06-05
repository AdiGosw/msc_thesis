# %%
import numpy as np
import random
from typing import List, Tuple, Dict
from scipy import stats as st
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
from joblib import Parallel, delayed
import time

# Global Seed
SEED = 67
np.random.seed(SEED)
random.seed(SEED)

# %%
class Environment:
    def __init__(self, v_high: float = 4.0, v_low: float = 0.0, mu: float = 0.5, sigma: float = 5.0):
        self.v_high = v_high
        self.v_low = v_low
        self.mu = mu    # Probability of the asset value being high
        self.sigma = sigma  # Liquidity shock of client
        self.current_asset_value = None
        self.reset()

    def reset(self):
        # Sample asset value
        self.current_asset_value = (self.v_high if random.random() < self.mu
                                  else self.v_low)
        return self.current_asset_value

    def step(self, prices: List[float]) -> Tuple[List[float], bool, dict]:
        """
        Execute one step of the market making game.
        No adverse selection: client draws their own independent value from
        the same binary distribution, independent of the realised asset value.
        """
        min_price = min(prices)
        num_best_offers = prices.count(min_price)

        # Client's independent value draw + liquidity shock
        liquidity_shock = np.random.normal(0, self.sigma)
        client_value     = (self.v_high if random.random() < self.mu else self.v_low)
        client_valuation = client_value + liquidity_shock

        # Check if client trades
        trade_occurred = client_valuation >= min_price

        # Calculate profits (market maker still trades at realised asset value cost)
        profits = []
        for price in prices:
            if trade_occurred and price == min_price:
                profit = (min_price - self.current_asset_value) / num_best_offers
            else:
                profit = 0.0
            profits.append(profit)

        info = {
            'asset_value': self.current_asset_value,
            'client_valuation': client_valuation,
            'min_price': min_price,
            'trade_occurred': trade_occurred
        }

        return profits, trade_occurred, info


# %%
class QAgent:
    def __init__(self, price_grid: List[float], alpha: float = 0.01,
                 beta: float = (8.1e-5), seed=None, initial_q: Tuple[float, float] = (3.0, 6.0),
                 stable_episodes_required: int = 50000):
        self.price_grid = price_grid
        self.num_prices = len(price_grid)
        self.alpha = alpha
        self.beta = beta
        self.stable_episodes_required = stable_episodes_required

        self.q_values = np.random.uniform(initial_q[0], initial_q[1], self.num_prices)
        self.episode = 0

        self.converged = False
        self.convergence_episode = None
        self.stable_action_episodes = 0
        self.last_greedy_action = None

        self.final_q_values = None
        self.final_greedy_action = None
        self.learning_complete = False

    def get_epsilon(self) -> float:
        return np.exp(-self.beta * self.episode)

    def choose_action_learning(self) -> int:
        epsilon = self.get_epsilon()
        if random.random() < epsilon:
            return random.randint(0, self.num_prices - 1)
        else:
            return np.argmax(self.q_values)

    def choose_action_exploitation(self) -> int:
        return self.final_greedy_action

    def update_q_value(self, price_index: int, profit: float):
        self.q_values[price_index] = (self.alpha * profit +
                                    (1 - self.alpha) * self.q_values[price_index])

    def check_convergence(self):
        if self.converged:
            return True

        current_greedy_action = np.argmax(self.q_values)

        if self.last_greedy_action is None:
            self.last_greedy_action = current_greedy_action
            self.stable_action_episodes = 1
        elif current_greedy_action == self.last_greedy_action:
            self.stable_action_episodes += 1
        else:
            self.stable_action_episodes = 1
            self.last_greedy_action = current_greedy_action

        if self.stable_action_episodes >= self.stable_episodes_required:
            self.converged = True
            self.convergence_episode = self.episode
            self.final_q_values = self.q_values.copy()
            self.final_greedy_action = current_greedy_action
            greedy_price = self.price_grid[current_greedy_action]
            print(f"Agent converged at episode {self.episode} (Greedy action stable for {self.stable_episodes_required} episodes, price: {greedy_price:.1f})")
            return True

        return False

    def finalize_learning(self):
        if not self.converged:
            self.final_q_values = self.q_values.copy()
            self.final_greedy_action = np.argmax(self.q_values)
        self.learning_complete = True
        print(f"Learning phase completed. Final greedy action: {self.final_greedy_action}, Price: {self.price_grid[self.final_greedy_action]:.1f}")

    def step_learning(self) -> Tuple[int, float]:
        if self.learning_complete:
            raise ValueError("Learning phase is complete. Use step_exploitation() instead.")
        action = self.choose_action_learning()
        price = self.price_grid[action]
        return action, price

    def step_exploitation(self) -> Tuple[int, float]:
        if not self.learning_complete:
            raise ValueError("Learning phase not complete. Use step_learning() instead.")
        action = self.choose_action_exploitation()
        price = self.price_grid[action]
        return action, price

    def learn(self, action: int, profit: float):
        if self.learning_complete:
            raise ValueError("Learning phase is complete. No more learning allowed.")
        self.update_q_value(action, profit)
        self.episode += 1
        self.check_convergence()

    def get_greedy_price(self) -> float:
        if self.learning_complete:
            return self.price_grid[self.final_greedy_action]
        else:
            greedy_action = np.argmax(self.q_values)
            return self.price_grid[greedy_action]

    def is_converged(self) -> bool:
        return self.converged

    def is_learning_complete(self) -> bool:
        return self.learning_complete

    def get_convergence_info(self) -> Dict:
        return {
            'converged': self.converged,
            'convergence_episode': self.convergence_episode,
            'current_episode': self.episode,
            'greedy_price': self.get_greedy_price(),
            'epsilon': self.get_epsilon() if not self.learning_complete else 0.0,
            'stable_action_episodes': self.stable_action_episodes,
            'greedy_action': self.final_greedy_action if self.learning_complete else np.argmax(self.q_values),
            'learning_complete': self.learning_complete
        }

    def get_learned_strategy(self) -> Dict:
        if not self.learning_complete:
            raise ValueError("Learning phase not complete yet.")
        return {
            'final_q_values': self.final_q_values.copy(),
            'final_greedy_action': self.final_greedy_action,
            'final_greedy_price': self.price_grid[self.final_greedy_action],
            'convergence_episode': self.convergence_episode,
            'total_learning_episodes': self.episode
        }


# %%
def run_single_experiment(experiment_id, track_every=1, seed=None, sigma=5.0):
    """Run a single experiment with separated learning and exploitation phases"""
    env = Environment(v_high=4.0, v_low=0.0, mu=0.5, sigma=sigma)
    price_grid = [round(p, 1) for p in np.arange(1.1, 14.9, 0.1)]

    agent1 = QAgent(price_grid, stable_episodes_required=50000)
    agent2 = QAgent(price_grid, stable_episodes_required=50000)

    price_history = []
    final_asset_value = None
    T = 500000

    for step in range(T):
        env.reset()
        current_asset_value = env.current_asset_value
        action1, price1 = agent1.step_learning()
        action2, price2 = agent2.step_learning()
        profits, trade_occurred, info = env.step([price1, price2])
        agent1.learn(action1, profits[0])
        agent2.learn(action2, profits[1])

        final_asset_value = current_asset_value

        if step % track_every == 0:
            price_history.append({
                'step': step,
                'greedy_price_1': agent1.get_greedy_price(),
                'greedy_price_2': agent2.get_greedy_price(),
                'min_greedy_price': min(agent1.get_greedy_price(), agent2.get_greedy_price())
            })

        if agent1.is_converged() and agent2.is_converged():
            learning_steps = step + 1
            break
    else:
        learning_steps = T

    final_min_price = min(agent1.get_greedy_price(), agent2.get_greedy_price())
    expected_asset_value = 2.0

    agent1.finalize_learning()
    agent2.finalize_learning()

    exploitation_results  = []
    exploitation_prices_1 = []
    exploitation_prices_2 = []

    for step in range(10000):
        asset_value = env.reset()
        action1, price1 = agent1.step_exploitation()
        action2, price2 = agent2.step_exploitation()
        profits, trade_occurred, info = env.step([price1, price2])

        exploitation_prices_1.append(price1)
        exploitation_prices_2.append(price2)

        exploitation_results.append({
            'prices':         [price1, price2],
            'profits':        profits,
            'min_price':      min(price1, price2),
            'asset_value':    asset_value,
            'trade_occurred': trade_occurred,
        })

    exploit_profits_1 = [r['profits'][0] for r in exploitation_results]
    exploit_profits_2 = [r['profits'][1] for r in exploitation_results]

    # QS_t = a^min_t - E[v~]  — unconditional over all T steps
    quoted_spreads = [r['min_price'] - expected_asset_value
                      for r in exploitation_results]
    # RS_t = a^min_t - v_t | trade occurred — conditional on client buying
    realized_spreads = [r['min_price'] - r['asset_value']
                        for r in exploitation_results if r['trade_occurred']]

    return {
        'experiment_id':         experiment_id,
        'learning_steps':        learning_steps,
        'converged':             agent1.is_converged() and agent2.is_converged(),
        'final_price_1':         agent1.get_greedy_price(),
        'final_price_2':         agent2.get_greedy_price(),
        'final_min_price':       final_min_price,
        'final_asset_value':     final_asset_value,
        'quoted_spread':         np.mean(quoted_spreads),
        'realized_spread':       np.mean(realized_spreads) if realized_spreads else 0.0,
        'avg_profit_1':          np.mean(exploit_profits_1),
        'avg_profit_2':          np.mean(exploit_profits_2),
        'price_history':         price_history,
        'exploitation_prices_1': exploitation_prices_1,
        'exploitation_prices_2': exploitation_prices_2,
    }


# %%
# Main execution output
if __name__ == "__main__":
    SIGMAS = [1, 3, 5, 7, 9]
    K      = 10
    N_JOBS = 10

    print(f"Running {len(SIGMAS) * K} total NAS experiments across sigmas {SIGMAS} with {N_JOBS} parallel jobs...")
    start_time = time.time()

    tasks = [(i, sigma) for sigma in SIGMAS for i in range(K)]

    all_results = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(run_single_experiment)(exp_id, 100, sigma=sigma)
        for exp_id, sigma in tasks
    )

    total_time = time.time() - start_time
    print(f"All experiments completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")

    for idx, sigma in enumerate(SIGMAS):
        results = all_results[idx * K : (idx + 1) * K]

        save_dict = {
            'quoted_spread':     [r['quoted_spread'] for r in results],
            'realized_spread':   [r['realized_spread'] for r in results],
            'exploit_price_1':   [r['exploitation_prices_1'] for r in results],
            'exploit_price_2':   [r['exploitation_prices_2'] for r in results],
            'final_prices_1':    [r['final_price_1'] for r in results],
            'final_prices_2':    [r['final_price_2'] for r in results],
            'final_min_price':   [r['final_min_price'] for r in results],
            'learning_steps': [r['learning_steps'] for r in results],
            'converged':         [r['converged'] for r in results],
            'avg_profit_1':      [r['avg_profit_1'] for r in results],
            'avg_profit_2':      [r['avg_profit_2'] for r in results],
            'price_history':     [r['price_history'] for r in results],
            'sigma':             sigma,
        }

        filename = f'qlearning_sigma{sigma}_N2_nas.npy'
        np.save(filename, save_dict)
        print(f"Saved results for sigma={sigma} to {filename}")
