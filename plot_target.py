import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

optims = {
    'E28A': np.genfromtxt("wfscores/E28A.txt", skip_header=1),
    'E28G': np.genfromtxt("wfscores/E28G.txt", skip_header=1),
    'U26A': np.genfromtxt("wfscores/US26A.txt", skip_header=1),
    'U26G': np.genfromtxt("wfscores/US26G.txt", skip_header=1),
}

# r - 0
# a - 1
# b - 2

xlims = [
    (1200, 1600),
    (1400, 2100),
    (1050, 1450),
    (1700, 2100)
]

datasets = {
    'Euro28_A': ('E28A', 81, 'Euro'),
    'Euro28_G': ('E28G', 81, 'Euro'),
    'US26_A': ('U26A', 83, 'US'),
    'US26_G': ('U26G', 83, 'US')
}

fig, ax = plt.subplots(len(datasets), 1, figsize=(8,8))

# Iterate datasets
for db_idx, filename in enumerate(datasets):
    # Load dataset and opts
    dbname, f, _ = datasets[filename]
    print(filename, dbname, f)

    ds = np.load("datasets/%s.npy" % dbname, allow_pickle=True)

    X = ds[:,:-2]
    c = ds[:,-2]
    y = ds[:,-1]
    opts = optims[dbname]

    # Get base solution
    base_solution = np.where(c==1)[0][-1]

    """
    Plot distributions
    """
    bw_method = 'silverman'

    maxval=0

    proba = np.linspace(*xlims[db_idx], 1000)

    """
    ax[db_idx].scatter(np.max(y[c==0]), 0, marker='x', c='black')
    ax[db_idx].scatter(np.max(y[c!=0]), 0, marker='x', c='green')
    ax[db_idx].scatter(np.max(opts[:,0]), 0, marker='x', c='blue')
    ax[db_idx].scatter(np.max(opts[:,1]), 0, marker='x', c='blue')
    ax[db_idx].scatter(np.max(opts[:,2]), 0, marker='x', c='red')
    ax[db_idx].scatter(np.max(opts[:,4]), 0, marker='x', c='red')
    ax[db_idx].scatter(np.max(opts[:,3]), 0, marker='x', c='red')
    """

    # Random
    ran_val = stats.gaussian_kde(y[c==0].tolist(),bw_method=bw_method)(proba)
    opt_val = stats.gaussian_kde(y[c!=0].tolist(),bw_method=bw_method)(proba)
    arc_val = stats.gaussian_kde(opts[:,0].tolist(),bw_method=bw_method)(proba)
    aroc_val = stats.gaussian_kde(opts[:,1].tolist(),bw_method=bw_method)(proba)
    a_val = stats.gaussian_kde(opts[:,2].tolist(),bw_method=bw_method)(proba)
    r_val = stats.gaussian_kde(opts[:,4].tolist(),bw_method=bw_method)(proba)
    o_val = stats.gaussian_kde(opts[:,3].tolist(),bw_method=bw_method)(proba)

    ax[db_idx].plot(proba, arc_val, c='blue', ls=":", label="ARC [%.0f]" % np.max(opts[:,0]), lw=1)
    ax[db_idx].plot(proba, aroc_val, c='blue', ls="--", label="AROC [%.0f]" % np.max(opts[:,1]), lw=1)
    ax[db_idx].plot(proba, ran_val, c='black', label="RAN [%.0f]" % np.max(y[c==0]), lw=1)
    ax[db_idx].plot(proba, opt_val, c='green', label="OPT [%.0f]" % np.max(y[c!=0]), lw=1)
    ax[db_idx].plot(proba, a_val, c='red', ls=":", label="A [%.0f]" % np.max(opts[:,2]), lw=1)
    ax[db_idx].plot(proba, r_val, c='red', ls="-", label="R [%.0f]" % np.max(opts[:,4]), lw=1)
    ax[db_idx].plot(proba, o_val, c='red', ls="--", label="O [%.0f]" % np.max(opts[:,3]), lw=1)

    # Calculate xlim
    ax[db_idx].grid(ls=":")
    ax[db_idx].set_title(dbname, fontsize=10)
    ax[db_idx].legend(frameon=False, fontsize=8, loc=2, ncol=4)
    ax[db_idx].set_xlabel("Accepted Traffic", fontsize=8)
    ax[db_idx].set_ylabel("kernel-density", fontsize=8)
    ax[db_idx].set_xlim(np.min(proba), np.max(proba))

    for a in [ax[db_idx]]:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)


    plt.tight_layout()
    plt.savefig("figures/target.png")
plt.savefig("figures/target.eps")
