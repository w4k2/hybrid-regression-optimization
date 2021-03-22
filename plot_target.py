import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

optims = {
    'E28A': np.genfromtxt("wfscores/E28A.txt", skip_header=1),
    'E28G': np.genfromtxt("wfscores/E28G.txt", skip_header=1),
    'U26A': np.genfromtxt("wfscores/US26A.txt", skip_header=1),
    'U26G': np.genfromtxt("wfscores/US26G.txt", skip_header=1),
}
q = optims['E28A'].shape[0]

print(q)

# r - 0
# a - 1
# b - 2

datasets = {
    'Euro28_A': ('E28A', 81, 'Euro'),
    'Euro28_G': ('E28G', 81, 'Euro'),
    'US26_A': ('U26A', 83, 'US'),
    'US26_G': ('U26G', 83, 'US')
}

fig, ax = plt.subplots(len(datasets), 2, figsize=(12,12))

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

    # Print random solutions
    ax[db_idx,0].scatter(y[c==0], c[c==0].astype(int),
                       c='black', s = 25, alpha=.25)

    # Print ILP and heuristics
    ax[db_idx,0].scatter(y[c!=0], c[c!=0].astype(int),
                       c='green', s = 25, alpha=.25)

    # Print base solution
    ax[db_idx,0].vlines(y[base_solution], -1, 8, color='black', ls=":", lw=1)

    # Print opts
    alphas = np.linspace(1,.1,q)
    for i in range(q):
        ax[db_idx,0].scatter(opts[i,0], 3, c='blue', alpha=.25)
        ax[db_idx,0].scatter(opts[i,1], 4, c='blue', alpha=.25)
        ax[db_idx,0].scatter(opts[i,2], 5, c='red', alpha=.25)
        ax[db_idx,0].scatter(opts[i,3], 6, c='red', alpha=.25)
        ax[db_idx,0].scatter(opts[i,4], 7, c='red', alpha=.25)

    # Set title
    ax[db_idx,0].set_title(filename, fontsize=10)

    """
    Plot distributions
    """
    mu = .1
    bw_method = 1

    maxval=0

    # Random
    density = stats.gaussian_kde(y[c==0].tolist(),bw_method=bw_method)
    proba = np.linspace(1000,2500, 100)
    val = density(proba)*mu
    ax[db_idx,1].plot(proba, val, c='black', label="random", lw=1)
    maxval = maxval if np.max(val) < maxval else np.max(val)

    # Heuristics and ILP
    density = stats.gaussian_kde(y[c!=0].tolist(),bw_method=bw_method)
    proba = np.linspace(1000,2500, 100)
    val = density(proba)*mu
    ax[db_idx,1].plot(proba, val, c='green', label="heuristics and ILP", lw=1)
    maxval = maxval if np.max(val) < maxval else np.max(val)

    # AWF
    density = stats.gaussian_kde(opts[:,0].tolist(),bw_method=bw_method)
    proba = np.linspace(1000,2500, 100)
    val = density(proba)*mu
    ax[db_idx,1].plot(proba, val, c='blue', ls=":", label="AWF", lw=1)
    maxval = maxval if np.max(val) < maxval else np.max(val)

    # BWF
    density = stats.gaussian_kde(opts[:,1].tolist(),bw_method=bw_method)
    proba = np.linspace(1000,2500, 100)
    val = density(proba)*mu
    ax[db_idx,1].plot(proba, val, c='blue', ls="--", label="BWF", lw=1)
    maxval = maxval if np.max(val) < maxval else np.max(val)

    # A
    density = stats.gaussian_kde(opts[:,2].tolist(),bw_method=bw_method)
    proba = np.linspace(1000,2500, 100)
    val = density(proba)*mu
    ax[db_idx,1].plot(proba, val, c='red', ls=":", label="A", lw=1)
    maxval = maxval if np.max(val) < maxval else np.max(val)

    # O
    density = stats.gaussian_kde(opts[:,3].tolist(),bw_method=bw_method)
    proba = np.linspace(1000,2500, 100)
    val = density(proba)*mu
    ax[db_idx,1].plot(proba, val, c='red', ls="--", label="O", lw=1)
    maxval = maxval if np.max(val) < maxval else np.max(val)

    # R
    density = stats.gaussian_kde(opts[:,4].tolist(),bw_method=bw_method)
    proba = np.linspace(1000,2500, 100)
    val = density(proba)*mu
    ax[db_idx,1].plot(proba, val, c='red', ls="-", label="R", lw=1)
    maxval = maxval if np.max(val) < maxval else np.max(val)


    maxval *= 1.1

    ax[db_idx,1].scatter([np.mean(y[c==0])], [maxval], c='black', marker='x')
    ax[db_idx,1].scatter([np.mean(y[c!=0])], [maxval], c='green', marker='x')
    ax[db_idx,1].scatter([np.mean(opts[:,0])], [maxval], c='blue', marker='x')
    ax[db_idx,1].scatter([np.mean(opts[:,1])], [maxval], c='blue', marker='x')
    ax[db_idx,1].scatter([np.mean(opts[:,2])], [maxval], c='red', marker='x')
    ax[db_idx,1].scatter([np.mean(opts[:,3])], [maxval], c='red', marker='x')
    ax[db_idx,1].scatter([np.mean(opts[:,4])], [maxval], c='red', marker='x')


    ax[db_idx,1].legend(frameon=False, fontsize=8)

    tax = ax[db_idx,0].twinx()
    tax.set_yticks([0,1,2,3,4,5,6,7])
    ax[db_idx,0].set_yticks([0,1,2,3,4,5,6,7])
    ax[db_idx,0].set_yticklabels(["random", "heuristics", "ILP", "AWF", "BWF", "A", "O", "R"], fontsize=8)
    tax.set_yticklabels([
        int(np.max(y[c==0])),
        int(np.max(y[c==1])),
        int(np.max(y[c==2])),
        int(np.max(opts[:,0])),
        int(np.max(opts[:,1])),
        int(np.max(opts[:,2])),
        int(np.max(opts[:,3])),
        int(np.max(opts[:,4]))], fontsize=8)
    ax[db_idx,0].grid(ls=":")
    ax[db_idx,0].set_ylim(-1,8)
    tax.set_ylim(-1,8)

    ax[db_idx,0].set_axisbelow(True)
    ax[db_idx,0].set_xlabel("Accepted Traffic", fontsize=8)
    ax[db_idx,1].set_xlabel("Accepted Traffic", fontsize=8)

    # Calculate xlim
    cent = y[base_solution]
    ma = np.max([np.max(y), np.max(opts)])
    dif = (ma-cent)*1.3
    ax[db_idx,0].set_xlim(cent-dif, cent+dif)
    ax[db_idx,1].set_xlim(cent-dif, cent+dif)
    ax[db_idx,1].set_title('Target function distribution', fontsize=10)
    ax[db_idx,1].set_yticks([])

    ax[db_idx,0].text(cent, 4, 'baseline', fontsize = 8, rotation=-90, va='center')

    for a in [ax[db_idx,0], ax[db_idx,1], tax]:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        a.spines['left'].set_visible(False)


    plt.tight_layout()
    plt.savefig("figures/target.png")
plt.savefig("figures/target.eps")
