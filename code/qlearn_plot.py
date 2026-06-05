import numpy as np

SIGMAS     = [1, 3, 5, 7, 9]
THESIS_DIR = 'C:/Users/adigo/Desktop/Thesis folder/Masters_Thesis'
unibonn_blue = '#004E9F'


def load_results(sigma):
    return np.load(f'qlearning_sigma{sigma}_N2.npy', allow_pickle=True).item()


def convergence_table(sigma=5):
    """
    Saves a LaTeX table of convergence statistics for a single sigma to THESIS_DIR.
    Columns: Experiment | Convergence Steps | Converged
    Plus a summary row with mean and std.
    """
    results   = load_results(sigma)
    steps     = np.array(results['learning_steps'])
    converged = np.array(results['converged'])

    tex = (
        r"\begin{table}[H]" + "\n"
        r"\centering" + "\n"
        f"\\caption{{Q-Learning Convergence Statistics}}" + "\n"
        r"\label{tab:ql_convergence}" + "\n"
        r"\begin{tabular}{cc}" + "\n"
        r"\hline" + "\n"
        r"Metric & Value \\" + "\n"
        r"\hline" + "\n"
        f"Mean Convergence Iterations & {steps.mean():,.0f} \\\\\n"
        f"Std Convergence Iterations  & {steps.std():,.0f} \\\\\n"
        f"Convergence Rate       & {converged.mean()*100:.0f}\\% \\\\\n"
        r"\hline" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}" + "\n"
    )

    print(tex)
    path = f'{THESIS_DIR}/ql_convergence_table.tex'
    with open(path, 'w') as f:
        f.write(tex)
    print(f"Saved to {path}")


if __name__ == "__main__":
    convergence_table()
