import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_ind

# Store datasets info
# r - 0
# a - 1
# b - 2
datasets = {
    'Euro28_A': ('E28A', 81, 'Euro'),
    'Euro28_G': ('E28G', 81, 'Euro'),
    'US26_A': ('U26A', 83, 'US'),
    'US26_G': ('U26G', 83, 'US')
}

fig, ax = plt.subplots(len(datasets), 1, figsize=(9,6))

# Iterate datasets
for db_idx, filename in enumerate(datasets):
    # Load dataset and opts
    dbname, f, _ = datasets[filename]

    ds = np.load("datasets/%s.npy" % dbname, allow_pickle=True)

    X = ds[:,:-2]
    c = ds[:,-2]
    y = ds[:,-1]

    # Get base solution
    base_solution_idx = np.where(c==1)[0][-1]

    random_solutions = y[c==0]
    ilp_solutions = y[c==1]
    heur_solutions = y[c==2]

    mean_ran = np.mean(random_solutions)
    mean_ilp = np.mean(ilp_solutions)
    mean_heu = np.mean(heur_solutions)

    # Baseline
    ax[db_idx].scatter([y[base_solution_idx]], [0], color='black', marker='x')

    """
    Plot distributions
    """
    proba = np.linspace(1000,2250, 1000)

    ran_den = stats.gaussian_kde(random_solutions.tolist(),
                                 bw_method='silverman')(proba)
    ran_heu = stats.gaussian_kde(ilp_solutions.tolist(),
                                 bw_method='silverman')(proba)
    ran_ilp = stats.gaussian_kde(heur_solutions.tolist(),
                                 bw_method='silverman')(proba)

    ax[db_idx].plot(proba, ran_den, c='black', label="RAN", lw=1)
    ax[db_idx].plot(proba, ran_heu, c='green', label="HEU", lw=1)
    ax[db_idx].plot(proba, ran_ilp, c='green', label="ILP", lw=1, ls=":")

    # Calculate xlim
    cent = y[base_solution_idx]
    ax[db_idx].grid(ls=":")
    ax[db_idx].set_title(dbname, fontsize=10)
    ax[db_idx].legend(frameon=False, fontsize=8)
    ax[db_idx].set_xlabel("Accepted Traffic", fontsize=8)
    ax[db_idx].set_ylabel("kernel-density", fontsize=8)
    ax[db_idx].set_xlim(np.min(proba), np.max(proba))

    for a in [ax[db_idx]]:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig("foo.png")
plt.savefig("figures/origins.eps")
