import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_ind

# Load calculated results for top Q
optims = {
    'E28A': np.genfromtxt("wfscores/E28A.txt", skip_header=1),
    'E28G': np.genfromtxt("wfscores/E28G.txt", skip_header=1),
    'U26A': np.genfromtxt("wfscores/US26A.txt", skip_header=1),
    'U26G': np.genfromtxt("wfscores/US26G.txt", skip_header=1),
}
q = optims['E28A'].shape[0]
alpha = .05
print("Q = %i optims" % q)

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

fig, ax = plt.subplots(len(datasets), 1, figsize=(12,12))

# Iterate datasets
for db_idx, filename in enumerate(datasets):
    # Load dataset and opts
    dbname, f, _ = datasets[filename]

    ds = np.load("datasets/%s.npy" % dbname, allow_pickle=True)

    X = ds[:,:-2]
    c = ds[:,-2]
    y = ds[:,-1]
    opts = optims[dbname][:q]

    print("\n# Dataset", filename, dbname, f, ds.shape, X.shape, c.shape, y.shape)

    # Get base solution
    base_solution_idx = np.where(c==1)[0][-1]
    print("Base solution - %5.2f - %i" % (y[base_solution_idx], base_solution_idx))

    random_solutions = y[c==0][:q]
    ilp_solutions = y[c==1][:q]
    heur_solutions = y[c==2][:q]

    awf_solutions = opts[:,0]
    bwf_solutions = opts[:,1]

    a_solutions = opts[:,2]
    o_solutions = opts[:,3]
    r_solutions = opts[:,4]

    ax[db_idx].vlines(y[base_solution_idx], -1, 8, color='black', ls=":", lw=1)

    """
    Plot distributions
    """
    mu = .1
    bw_method = 2
    maxval = 0

    # Random
    density = stats.gaussian_kde(y[c==0].tolist(),bw_method=bw_method)
    proba = np.linspace(500,2500, 1000)
    val = density(proba)*mu
    ax[db_idx].plot(proba, val, c='black', label="RAN", lw=1)
    maxval = maxval if np.max(val) < maxval else np.max(val)

    # Heuristics
    density = stats.gaussian_kde(y[c==1].tolist(),bw_method=bw_method)
    proba = np.linspace(500,2500, 1000)
    val = density(proba)*mu
    ax[db_idx].plot(proba, val, c='green', label="HEU", lw=1)
    maxval = maxval if np.max(val) < maxval else np.max(val)

    # ILP
    density = stats.gaussian_kde(y[c==2].tolist(),bw_method=bw_method)
    proba = np.linspace(500,2500, 1000)
    val = density(proba)*mu
    ax[db_idx].plot(proba, val, c='green', label="ILP", lw=1, ls=":")
    maxval = maxval if np.max(val) < maxval else np.max(val)

    maxval *= 1.1

    ax[db_idx].scatter([np.mean(y[c==0])], [maxval], c='black', marker='x')
    ax[db_idx].scatter([np.mean(y[c==1])], [maxval], c='green', marker='x')
    ax[db_idx].scatter([np.mean(y[c==2])], [maxval], c='green', marker='x')

    ax[db_idx].legend(frameon=False, fontsize=8)

    ax[db_idx].set_xlabel("Accepted Traffic", fontsize=8)

    ax[db_idx].set_ylim(0,maxval*1.1)

    # Calculate xlim
    cent = y[base_solution_idx]
    ma = np.max([np.max(y), np.max(opts)])
    dif = (ma-cent)*1.3
    #ax[db_idx].set_xlim(cent-dif, cent+dif)
    ax[db_idx].set_title('Target function distribution', fontsize=10)
    ax[db_idx].set_yticks([])

    for a in [ax[db_idx]]:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        a.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig("figures/origins.png")
plt.savefig("figures/origins.eps")
