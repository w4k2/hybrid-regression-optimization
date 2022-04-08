"""
Ploty dystrybucji cech.
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# np.set_printoptions(precision=3)

datasets = {
    'Euro28_A': ('E28A', 81, 'Euro'),
    'Euro28_G': ('E28G', 81, 'Euro'),
    'US26_A': ('U26A', 83, 'US'),
    'US26_G': ('U26G', 83, 'US')
}

# Iterujemy zbiory danych
for filename in datasets:
    # Load dataset
    dbname, f, kao = datasets[filename] # kao - identyfikator kangurka
    ds = np.load("datasets/%s.npy" % dbname, allow_pickle=True)
    X = ds[:,:-2]               # dane
    c = ds[:,-2].astype(int)    # kategorie
    y = ds[:,-1]                # etykiety
    X_kao = np.load('kangurki/%s.npy' % kao) # kangurki

    print(dbname)

    # prawie kwadrat
    sub_len = int(np.ceil(np.sqrt(X_kao.shape[1])))
    fig, axs = plt.subplots(sub_len, sub_len, figsize=(16, 16))
    axs = iter(axs.flatten())

    for x_val, ax in zip(X.T, axs):
        # density = stats.gaussian_kde(x_val.tolist(), bw_method='silverman')
        # proba = np.linspace(1000,2500,1000)
        # val = density(proba)

        x_val = x_val.astype(np.int)
        y = np.arange(11)

        bins = np.bincount(x_val[c==0], minlength=11)
        bins = bins / bins.sum()
        ax.plot(y, bins, c='black', ls=':', lw=1)

        bins = np.bincount(x_val[c==1], minlength=11)
        bins = bins / bins.sum()
        ax.plot(y, bins, c='green', ls='-', lw=1)

        bins = np.bincount(x_val[c==2], minlength=11)
        bins = bins / bins.sum()
        ax.plot(y, bins, c='blue', ls='-', lw=1)

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # print(bins)
        # exit()

    # a na co to komu?
    for ax in axs:
        ax.axis('off')

    plt.savefig('figures/attdist_%s.png' % dbname)
    # exit()
