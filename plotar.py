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

fig, ax = plt.subplots(len(datasets), 2, figsize=(12,12))

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

    solvector = [
        random_solutions, ilp_solutions, heur_solutions,
        awf_solutions, bwf_solutions,
        a_solutions, o_solutions, r_solutions
    ]

    is_dependent = (np.array([[ttest_ind(solvector[i], solvector[j]).pvalue
                               for i in range(len(solvector))]
                              for j in range(len(solvector))]) > alpha).astype(int)

    is_higher = (np.array([[np.mean(solvector[i]) < np.mean(solvector[j])
                               for i in range(len(solvector))]
                              for j in range(len(solvector))])).astype(int)

    #print(is_higher)

    is_better = (1-is_dependent) * is_higher
    #print(is_better)

    is_dependent = is_better

    # print(is_dependent)

    tab_s = []
    tab_d = []
    tab_dep = []

    tab_s.append(y[base_solution_idx])
    tab_d.append(0)
    tab_dep.append([])

    #print("\nRAN solutions - %5.2f - %5.2f\t" %
    #    (np.mean(random_solutions),
    #     np.std(random_solutions)
    #    ), is_dependent[0], len(random_solutions)
    #)
    tab_s.append(np.mean(random_solutions))
    tab_d.append(np.std(random_solutions))
    tab_dep.append(np.where(is_dependent[0]==1)[0]+1)

    #print("HEU solutions - %5.2f - %5.2f\t" %
    #    (np.mean(ilp_solutions),
    #    np.std(ilp_solutions)
    #    ), is_dependent[1], len(ilp_solutions)
    #)
    tab_s.append(np.mean(ilp_solutions))
    tab_d.append(np.std(ilp_solutions))
    tab_dep.append(np.where(is_dependent[1]==1)[0]+1)

    #print("ILP solutions - %5.2f - %5.2f\t" %
    #    (np.mean(heur_solutions),
    #    np.std(heur_solutions)
    #    ), is_dependent[2], len(heur_solutions)
    #)
    tab_s.append(np.mean(heur_solutions))
    tab_d.append(np.std(heur_solutions))
    tab_dep.append(np.where(is_dependent[2]==1)[0]+1)

    #print("AWF solutions - %5.2f - %5.2f\t" %
    #    (np.mean(awf_solutions),
    #    np.std(awf_solutions)
    #    ), is_dependent[3], len(awf_solutions)
    #)
    tab_s.append(np.mean(awf_solutions))
    tab_d.append(np.std(awf_solutions))
    tab_dep.append(np.where(is_dependent[3]==1)[0]+1)

    #print("BWF solutions - %5.2f - %5.2f\t" %
    #    (np.mean(bwf_solutions),
    #    np.std(bwf_solutions)
    #    ), is_dependent[4], len(bwf_solutions)
    #)
    tab_s.append(np.mean(bwf_solutions))
    tab_d.append(np.std(bwf_solutions))
    tab_dep.append(np.where(is_dependent[4]==1)[0]+1)

    #print("  A solutions - %5.2f - %5.2f\t" %
    #    (np.mean(a_solutions),
    #    np.std(a_solutions)
    #    ), is_dependent[5], len(a_solutions)
    #)
    tab_s.append(np.mean(a_solutions))
    tab_d.append(np.std(a_solutions))
    tab_dep.append(np.where(is_dependent[5]==1)[0]+1)

    #print("  O solutions - %5.2f - %5.2f\t" %
    #    (np.mean(o_solutions),
    #    np.std(o_solutions)
    #    ), is_dependent[6],len(o_solutions)
    #)
    tab_s.append(np.mean(o_solutions))
    tab_d.append(np.std(o_solutions))
    tab_dep.append(np.where(is_dependent[6]==1)[0]+1)

    #print("  R solutions - %5.2f - %5.2f\t" %
    #    (np.mean(r_solutions),
    #    np.std(r_solutions)
    #    ), is_dependent[7], len(r_solutions)
    #)
    tab_s.append(np.mean(r_solutions))
    tab_d.append(np.std(r_solutions))
    tab_dep.append(np.where(is_dependent[7]==1)[0]+1)

    tab_s = np.array(tab_s)
    tab_d = np.array(tab_d)
    # print(tab_s)
    # print(tab_d)
    # print(tab_dep)

    print(' & '.join(["%.0f" % _ for _ in tab_s]), "\\\\")
    print(' & '.join(["%.2f" % _ for _ in tab_d]), "\\\\")
    print(' & '.join([', '.join(["%i" % __ for __ in _]) for _ in tab_dep]), "\\\\")


    #continue

    # Print random solutions
    ax[db_idx,0].scatter(y[c==0], c[c==0].astype(int),
                       c='black', s = 25, alpha=.25)

    # Print ILP and heuristics
    ax[db_idx,0].scatter(y[c!=0], c[c!=0].astype(int),
                       c='green', s = 25, alpha=.25)

    # Print base solution
    ax[db_idx,0].vlines(y[base_solution_idx], -1, 8, color='black', ls=":", lw=1)
    ax[db_idx,1].vlines(y[base_solution_idx], -1, 8, color='black', ls=":", lw=1)

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
    mu = .001
    bw_method = 2
    maxval = 0

    # Random
    density = stats.gaussian_kde(y[c==0].tolist(),bw_method=bw_method)
    proba = np.linspace(1000,2500, 100)
    val = density(proba)*mu
    ax[db_idx,1].plot(proba, val, c='black', label="RAN", lw=1)
    maxval = maxval if np.max(val) < maxval else np.max(val)


    # Heuristics and ILP
    density = stats.gaussian_kde(y[c==1].tolist(),bw_method=bw_method)
    proba = np.linspace(1000,2500, 100)
    val = density(proba)*mu
    ax[db_idx,1].plot(proba, val, c='green', label="HEU", lw=1)
    maxval = maxval if np.max(val) < maxval else np.max(val)

    # Heuristics and ILP
    density = stats.gaussian_kde(y[c==2].tolist(),bw_method=bw_method)
    proba = np.linspace(1000,2500, 100)
    val = density(proba)*mu
    ax[db_idx,1].plot(proba, val, c='green', label="ILP", lw=1, ls=":")
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
    ax[db_idx,0].set_yticklabels(["RAN", "HEU", "ILP", "AWF", "BWF", "A", "O", "R"], fontsize=8)
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

    ax[db_idx,1].set_ylim(0,maxval*1.1)

    # Calculate xlim
    cent = y[base_solution_idx]
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
