import numpy as np
import matplotlib.pyplot as plt

SIGMAS     = [1, 3, 5, 7, 9]
THESIS_DIR = 'C:/Users/adigo/Desktop/Thesis folder/Masters_Thesis'
unibonn_blue = '#004E9F'


def load_results(sigma, run=9):
    """Load a single run from the new list-of-runs format."""
    data = np.load(f"results_sigma{sigma}_N2.npy", allow_pickle=True)
    runs = list(data) if data.ndim != 0 else [data.item()]
    return runs[run]


def smooth(arr, w):
    """Causal expanding-window rolling mean — same length as input, starts from index 0."""
    out = np.empty(len(arr), dtype=float)
    for i in range(len(arr)):
        out[i] = np.mean(arr[max(0, i - w + 1): i + 1])
    return out


def plot_loss(results, sigma, window=1000):
    actor_losses  = np.array(results['agent1']['actor_losses'])
    critic_losses = np.array(results['agent1']['critic_losses'])

    iterations = np.arange(1, len(actor_losses) + 1)

    actor_sm  = smooth(actor_losses,  window)
    critic_sm = smooth(critic_losses, window)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Training Loss ($\\sigma$={sigma})', fontsize=13)

    ax1.plot(iterations, actor_losses, alpha=0.15, color=unibonn_blue)
    ax1.plot(iterations, actor_sm, color=unibonn_blue, linewidth=2)
    ax1.set_title('Panel A: Actor Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Actor Loss')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(iterations, critic_losses, alpha=0.15, color=unibonn_blue)
    ax2.plot(iterations, critic_sm, color=unibonn_blue, linewidth=2)
    ax2.set_title('Panel B: Critic Loss')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Critic Loss')
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{THESIS_DIR}/loss_sigma{sigma}.pdf', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    results = load_results(sigma=5)
    plot_loss(results, sigma=5)
