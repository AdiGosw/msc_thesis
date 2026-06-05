# %%
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import List, Tuple, Dict
from scipy.optimize import brentq
from scipy.stats import norm
import time
import json
from multiprocessing import Pool

SEED = 67

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %%
class Environment:
    def __init__(self, v_high: float = 4.0, v_low: float = 0.0, mu: float = 0.5, sigma: float = 5.0):
        self.v_high = v_high
        self.v_low = v_low
        self.mu = mu  # Probability of the asset value being high
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
        Execute one step of the market making game
        """
        min_price = min(prices)
        num_best_offers = prices.count(min_price)
        
        # Sample client's liquidity shock
        liquidity_shock = np.random.normal(0, self.sigma)
        client_valuation = self.current_asset_value + liquidity_shock
        
        # Check if client trades
        trade_occurred = client_valuation >= min_price
        
        # Calculate profits
        profits = []
        for price in prices:
            if trade_occurred and price == min_price:
                # Share profit equally among best offers
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
class DDPGParams:
    """Configuration class for DDPG hyperparameters"""
    def __init__(self, 
                 # Network architecture
                 actor_hidden_dim=64,
                 critic_hidden_dim=32,
                 
                 # Learning parameters
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 gamma=0,
                 tau=0.001,
                 
                 # Exploration with exponential decay
                 initial_noise_std=0.5,
                 final_noise_std=0.01,
                 noise_decay_episodes=1386,

                 # Training
                 batch_size=500,
                 buffer_capacity=100000,

                 # Episode settings
                 max_episodes=1540,
                 max_steps_per_episode=100,
                 
                 # Action bounds
                 min_action=1.1,
                 max_action=14.9,
                 
                 # Device
                 device=None):
        
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.initial_noise_std = initial_noise_std
        self.final_noise_std = final_noise_std
        self.noise_decay_episodes = noise_decay_episodes
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.min_action = min_action
        self.max_action = max_action
        self.device = device if device else 'cpu'
    
    def get_noise_std(self, episode):
        """Calculate exponentially decaying noise standard deviation"""
        if episode >= self.noise_decay_episodes:
            return self.final_noise_std
        
        # Exponential decay: noise = initial * exp(-decay_rate * episode)
        decay_rate = -np.log(self.final_noise_std / self.initial_noise_std) / self.noise_decay_episodes
        noise_std = self.initial_noise_std * np.exp(-decay_rate * episode)
        return max(noise_std, self.final_noise_std)
    
    def __str__(self):
        """Print all parameters"""
        return f"""
DDPG Parameters:
  Network: actor_dim={self.actor_hidden_dim}, critic_dim={self.critic_hidden_dim}
  Learning: actor_lr={self.actor_lr} critic_lr ={self.critic_lr}, gamma={self.gamma}, tau={self.tau}
  Exploration: initial_noise={self.initial_noise_std}, final_noise={self.final_noise_std}, decay_episodes={self.noise_decay_episodes}
  Training: batch_size={self.batch_size}, buffer_capacity={self.buffer_capacity}
  Episodes: max_episodes={self.max_episodes}, max_steps={self.max_steps_per_episode}
  Action bounds: [{self.min_action}, {self.max_action}]
  Device: {self.device}
        """

class Actor(nn.Module):
    """Actor network for DDPG with separate hidden dimensions"""
    def __init__(self, state_dim, action_dim, params):
        super(Actor, self).__init__()
        self.min_action = params.min_action
        self.max_action = params.max_action
        
        # Network
        self.fc1 = nn.Linear(state_dim, params.actor_hidden_dim)
        self.fc2 = nn.Linear(params.actor_hidden_dim, params.actor_hidden_dim)
        self.fc3 = nn.Linear(params.actor_hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        
        # Custom action scaling
        mid = 0.5 * (self.min_action + self.max_action)
        half = 0.5 * (self.max_action - self.min_action)
        
        out = mid + half * x
        return out

class Critic(nn.Module):
    """Critic network for DDPG with 64 hidden dimensions"""
    def __init__(self, state_dim, action_dim, params):
        super(Critic, self).__init__()
        
        # Larger network with 64 hidden units
        self.fc1 = nn.Linear(state_dim + action_dim, params.critic_hidden_dim)
        self.fc2 = nn.Linear(params.critic_hidden_dim, params.critic_hidden_dim)
        self.fc3 = nn.Linear(params.critic_hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ReplayBuffer:
    def __init__(self, params):
        self.buffer = deque(maxlen=params.buffer_capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
      
class DDPGAgent:
    """DDPG Agent with exponential noise decay"""
    def __init__(self, state_dim, action_dim, params):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.params = params
        self.current_episode = 0
        
        # Networks
        self.actor = Actor(state_dim, action_dim, params).to(params.device)
        self.actor_target = Actor(state_dim, action_dim, params).to(params.device)
        self.critic = Critic(state_dim, action_dim, params).to(params.device)
        self.critic_target = Critic(state_dim, action_dim, params).to(params.device)
        
        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=params.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=params.critic_lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(params)
        
        self.actor_losses  = []
        self.critic_losses = []
        
    def select_action(self, state, add_noise=True):
        """Select action with exponentially decaying noise"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.params.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()

        if add_noise:
            current_noise_std = self.params.get_noise_std(self.current_episode)
            action = action + np.random.normal(0, current_noise_std, size=self.action_dim)
            
        # Clip action to valid bounds
        action = np.clip(action, self.params.min_action, self.params.max_action)
        return action[0]
    
    def start_new_episode(self):
        """Call this at the start of each episode to update noise decay"""
        self.current_episode += 1
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, [action], reward, next_state, done)
    
    def update(self):
        """Update actor and critic networks"""
        if len(self.replay_buffer) < self.params.batch_size:
            return None, None
        
        # Sample batch
        state, action, reward, next_state, done = self.replay_buffer.sample(self.params.batch_size)
        
        state = torch.FloatTensor(state).to(self.params.device)
        action = torch.FloatTensor(action).to(self.params.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.params.device)
        next_state = torch.FloatTensor(next_state).to(self.params.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.params.device)
        
        # Update Critic
        next_action = self.actor_target(next_state)
        target_q = self.critic_target(next_state, next_action)
        target_q = reward + (self.params.gamma * target_q * (1 - done))
        
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.actor_target, self.actor, self.params.tau)
        self.soft_update(self.critic_target, self.critic, self.params.tau)
        
        # Track losses
        actor_loss_value = actor_loss.item()
        critic_loss_value = critic_loss.item()
        self.actor_losses.append(actor_loss_value)
        self.critic_losses.append(critic_loss_value)
        
        return actor_loss_value, critic_loss_value
    
    def soft_update(self, target, source, tau):
        """Soft update target networks"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def _fmt(x):
    return str(int(x)) if float(x) == int(x) else str(x)

# %%
def train_single_market_ddpg(params=None, sigma=5.0, delta_v=4.0, N=2, save=True):
    """Train N DDPG agents competing in one market environment.

    Args:
        params:   DDPGParams instance (uses defaults if None).
        sigma:    Std. dev. of the client's liquidity shock.
        delta_v:  Asset value range; v_high = delta_v, v_low = 0.
        N:        Number of competing market makers.
    """
    if params is None:
        params = DDPGParams()

    print(f"Training {N} DDPG agents in single market "
          f"(sigma={sigma}, delta_v={delta_v}, N={N})...")
    print(params)

    env = Environment(v_high=delta_v, v_low=0.0, sigma=sigma)

    state_dim  = 1
    action_dim = 1

    agents = [DDPGAgent(state_dim, action_dim, params) for _ in range(N)]

    # Give each agent a different initial price drawn uniformly from the full
    # action range. Only the fc3 bias is adjusted; weights stay at PyTorch
    # defaults so the network remains fully expressive from the start.
    # output = mid + half * tanh(fc3(x))  →  pre_tanh = arctanh((price - mid) / half)
    mid  = 0.5 * (params.min_action + params.max_action)
    half = 0.5 * (params.max_action - params.min_action)
    for agent in agents:
        init_price = random.uniform(3.0, 6.0)
        pre_tanh   = float(np.arctanh(np.clip((init_price - mid) / half, -0.999, 0.999)))
        with torch.no_grad():
            agent.actor.fc3.bias.fill_(pre_tanh)
            agent.actor_target.fc3.bias.fill_(pre_tanh)

    episode_rewards = [[] for _ in range(N)]
    episode_prices  = [[] for _ in range(N)]
    min_prices      = []

    start_time = time.time()

    # ── Learning Phase ──────────────────────────────────────────
    for episode in range(params.max_episodes):
        for agent in agents:
            agent.start_new_episode()

        ep_rewards = [0.0] * N

        for step in range(params.max_steps_per_episode):
            env.reset()
            state = np.array([1.0])

            actions = [agent.select_action(state, add_noise=True) for agent in agents]
            profits, trade_occurred, info = env.step(actions)

            next_state = np.array([1.0])

            update_order = list(range(N))
            random.shuffle(update_order)
            for i in update_order:
                agents[i].store_transition(state, actions[i], profits[i], next_state, True)
                agents[i].update()
                ep_rewards[i] += profits[i]

        # Greedy price snapshot: query actor deterministically at end of episode
        greedy_state   = np.array([1.0])
        greedy_actions = [agent.select_action(greedy_state, add_noise=False) for agent in agents]
        for i in range(N):
            episode_rewards[i].append(ep_rewards[i])
            episode_prices[i].append(greedy_actions[i])
        min_prices.append(min(greedy_actions))

        if (episode + 1) % 100 == 0:
            avg_prices  = [np.mean(episode_prices[i][-20:])  for i in range(N)]
            avg_rewards = [np.mean(episode_rewards[i][-20:]) for i in range(N)]
            status = ", ".join(
                f"Agent{i+1} Price={avg_prices[i]:.2f} Reward={avg_rewards[i]:.2f}"
                for i in range(N)
            )
            print(f"Episode {episode + 1}/{params.max_episodes}: {status}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f}s")

    # ── Exploitation Phase ──────────────────────────────────────
    print("\nRunning exploitation phase...")
    exploitation_results = []
    EXPLOITATION_STEPS = 10000

    for _ in range(EXPLOITATION_STEPS):
        asset_value = env.reset()
        state = np.array([1.0])  # uninformed: agent does not observe v_t

        actions = [agent.select_action(state, add_noise=False) for agent in agents]
        profits, trade_occurred, info = env.step(actions)

        record = {
            'min_price':      min(actions),
            'asset_value':    asset_value,
            'trade_occurred': trade_occurred,
            'prices':         actions,
            'profits':        profits,
        }
        for i, (a, p) in enumerate(zip(actions, profits)):
            record[f'price_{i+1}']  = a
            record[f'profit_{i+1}'] = p
        exploitation_results.append(record)

    # ── Spreads ─────────────────────────────────────────────────
    expected_asset_value = env.mu * env.v_high + (1 - env.mu) * env.v_low  # E[v~] = 2.0

    # QS_t = a^min_t - E[v~]  — unconditional over all T steps
    quoted_spreads   = [r['min_price'] - expected_asset_value
                        for r in exploitation_results]
    # RS_t = a^min_t - v_t | trade occurred — conditional on client buying
    realized_spreads = [r['min_price'] - r['asset_value']
                        for r in exploitation_results if r['trade_occurred']]

    agents_out = [
        {
            'episode_rewards': episode_rewards[i],
            'episode_prices':  episode_prices[i],
            'final_price':     np.mean(episode_prices[i][-10:]),
            'actor_losses':    agents[i].actor_losses,
            'critic_losses':   agents[i].critic_losses,
        }
        for i in range(N)
    ]

    result = {
        'agents':             agents_out,
        'min_prices':         min_prices,
        'quoted_spreads':     quoted_spreads,
        'realized_spreads':   realized_spreads,
        'exploitation':       exploitation_results,
    }
    # Backward-compatible per-agent keys (agent1, agent2, …)
    for i, ag in enumerate(agents_out):
        result[f'agent{i+1}'] = ag

    if save:
        filename = f"results_sigma{_fmt(sigma)}_N{N}.npy"
        np.save(filename, result)
        print(f"Results saved to {filename}")

    return result

# %%
N_JOBS = 7  # 13 runs in parallel across all conditions

def _run_condition(label, sigma, delta_v, N, run_id=0):
    seed = SEED + run_id
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    threads_per_worker = max(1, os.cpu_count() // N_JOBS)
    torch.set_num_threads(threads_per_worker)
    print(f"Starting run {run_id}: {label} (sigma={sigma}, delta_v={delta_v}, N={N})")
    result = train_single_market_ddpg(sigma=sigma, delta_v=delta_v, N=N, save=False)
    return label, run_id, result


def _run_task(args):
    return _run_condition(*args)


def run_all_conditions(n_runs=10):
    conditions = [
        {'label': 'sigma=5', 'sigma': 5, 'delta_v': 4, 'N': 2},
        {'label': 'sigma=1', 'sigma': 1, 'delta_v': 4, 'N': 2},
        {'label': 'sigma=3', 'sigma': 3, 'delta_v': 4, 'N': 2},
        {'label': 'sigma=7', 'sigma': 7, 'delta_v': 4, 'N': 2},
        {'label': 'sigma=9', 'sigma': 9, 'delta_v': 4, 'N': 2},
    ]

    task_args = [(c['label'], c['sigma'], c['delta_v'], c['N'], run_id)
                 for c in conditions for run_id in range(n_runs)]

    raw = []
    with Pool(processes=N_JOBS) as pool:
        for i, result in enumerate(pool.imap_unordered(_run_task, task_args), 1):
            raw.append(result)
            label_done, run_id_done, _ = result
            print(f"[{i:2d}/{len(task_args)}] Done: {label_done} run {run_id_done}", flush=True)

    # Group results by label, preserving run order
    grouped = {c['label']: [] for c in conditions}
    for label, run_id, result in raw:
        grouped[label].append((run_id, result))
    for label in grouped:
        grouped[label] = [r for _, r in sorted(grouped[label])]

    # Save each condition as a list of n_runs results (same format as Q-learning)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for c in conditions:
        label  = c['label']
        sigma  = c['sigma']
        N      = c['N']
        filename = os.path.join(script_dir, f"results_sigma{_fmt(sigma)}_N{N}.npy")
        np.save(filename, grouped[label])
        print(f"Saved {n_runs} runs for {label} to {filename}")

    return grouped

# %%
# Main execution
if __name__ == "__main__":
    all_results = run_all_conditions()
    # all_results: {label: [list of n_runs result dicts]}
    results = all_results['sigma=5']   # list of 10 runs

    print(f"\n{'='*76}")
    print(f"{'SUMMARY':^76}")
    print(f"{'='*76}")
    print(f"{'Condition':<20} {'Agent 1 Price':>13} {'Agent 2 Price':>13} {'Mean QS':>12} {'Mean RS':>12}")
    print(f"{'-'*76}")
    for lbl, runs in all_results.items():
        exploit = [r for run in runs for r in run['exploitation']]
        price_1 = np.mean([r['price_1'] for r in exploit])
        price_2 = np.mean([r['price_2'] for r in exploit])
        mean_qs = np.mean([s for run in runs for s in run['quoted_spreads']])
        mean_rs = np.mean([s for run in runs for s in run['realized_spreads']])
        print(f"{lbl:<20} {price_1:>13.3f} {price_2:>13.3f} {mean_qs:>12.3f} {mean_rs:>12.3f}")
    print(f"{'='*76}")

    r0 = results[0]
    print(f"\nFinal Results (sigma=5, run 0):")
    print(f"Agent 1 final price: {r0['agent1']['final_price']:.3f}")
    print(f"Agent 2 final price: {r0['agent2']['final_price']:.3f}")
    print(f"Best offer: {min(r0['agent1']['final_price'], r0['agent2']['final_price']):.3f}")
    print(f"Agent 1 - Mean: {np.mean(r0['agent1']['episode_prices'][-50:]):.3f}, "
          f"Median: {np.median(r0['agent1']['episode_prices'][-50:]):.3f}, "
          f"Std: {np.std(r0['agent1']['episode_prices'][-50:]):.3f}")
    print(f"Agent 2 - Mean: {np.mean(r0['agent2']['episode_prices'][-50:]):.3f}, "
          f"Median: {np.median(r0['agent2']['episode_prices'][-50:]):.3f}, "
          f"Std: {np.std(r0['agent2']['episode_prices'][-50:]):.3f}")

    def statistics(res):
        print("Agent price statistics - full training phase")
        print(f"Agent 1 - Mean: {np.mean(res['agent1']['episode_prices']):.3f}, "
              f"Median: {np.median(res['agent1']['episode_prices']):.3f}, "
              f"Std: {np.std(res['agent1']['episode_prices']):.3f}")
        print(f"Agent 2 - Mean: {np.mean(res['agent2']['episode_prices']):.3f}, "
              f"Median: {np.median(res['agent2']['episode_prices']):.3f}, "
              f"Std: {np.std(res['agent2']['episode_prices']):.3f}")
        print("\nAgent price statistics - last 50 training episodes")
        print(f"Agent 1 - Mean: {np.mean(res['agent1']['episode_prices'][-50:]):.3f}, "
              f"Median: {np.median(res['agent1']['episode_prices'][-50:]):.3f}, "
              f"Std: {np.std(res['agent1']['episode_prices'][-50:]):.3f}")
        print(f"Agent 2 - Mean: {np.mean(res['agent2']['episode_prices'][-50:]):.3f}, "
              f"Median: {np.median(res['agent2']['episode_prices'][-50:]):.3f}, "
              f"Std: {np.std(res['agent2']['episode_prices'][-50:]):.3f}")
        exploit_prices_1 = [r['price_1'] for r in res['exploitation']]
        exploit_prices_2 = [r['price_2'] for r in res['exploitation']]
        print("\nAgent price statistics - exploitation phase (no noise)")
        print(f"Agent 1 - Mean: {np.mean(exploit_prices_1):.3f}, "
              f"Median: {np.median(exploit_prices_1):.3f}, "
              f"Std: {np.std(exploit_prices_1):.3f}")
        print(f"Agent 2 - Mean: {np.mean(exploit_prices_2):.3f}, "
              f"Median: {np.median(exploit_prices_2):.3f}, "
              f"Std: {np.std(exploit_prices_2):.3f}")
        print(f"Mean Quoted Spread:       {np.mean(res['quoted_spreads']):.3f}")
        print(f"Std. Dev Quoted Spread:   {np.std(res['quoted_spreads']):.3f}")
        print(f"Mean Realized Spread:     {np.mean(res['realized_spreads']):.3f}")
        print(f"Std. Dev Realized Spread: {np.std(res['realized_spreads']):.3f}")
        print(f"Mean Final Min Price:     {np.mean(res['min_prices']):.3f}")
        print(f"Std. Dev Final Price:     {np.std(res['min_prices']):.3f}")

    statistics(results[0])

    # Aggregate exploitation data across all 10 runs
    exploit = [r for run in results for r in run['exploitation']]
    all_qs  = [s for run in results for s in run['quoted_spreads']]
    all_rs  = [s for run in results for s in run['realized_spreads']]

    ddpg_stats = {
        'agent1_mean':      float(np.mean([r['price_1'] for r in exploit])),
        'agent1_std':       float(np.std([r['price_1'] for r in exploit])),
        'agent2_mean':      float(np.mean([r['price_2'] for r in exploit])),
        'agent2_std':       float(np.std([r['price_2'] for r in exploit])),
        'min_price_mean':   float(np.mean([r['min_price'] for r in exploit])),
        'min_price_std':    float(np.std([r['min_price'] for r in exploit])),
        'quoted_mean':      float(np.mean(all_qs)),
        'quoted_std':       float(np.std(all_qs)),
        'realized_mean':    float(np.mean(all_rs)),
        'realized_std':     float(np.std(all_rs)),
        'final_price_1':    float(results[0]['agent1']['final_price']),
        'final_price_2':    float(results[0]['agent2']['final_price']),
    }

    with open('C:/Users/adigo/Desktop/Thesis folder/Masters_Thesis/ddpg_stats.json', 'w') as f:
        json.dump(ddpg_stats, f)

    print("DDPG stats saved")


# # %%
# import subprocess
# subprocess.run(["C:/Users/adigo/anaconda3/python.exe", "-m", "pip", "install", "jupytext"])

# # %%
# import jupytext
# nb = jupytext.read("c:/Users/adigo/Desktop/Thesis folder/code/DDPG.py")
# jupytext.write(nb, "c:/Users/adigo/Desktop/Thesis folder/code/DDPG2.ipynb")

# %%
