"""
Weryfikacja jakości modelu regresji w predykcji rozwiazan z optymalizatorow.
"""
import numpy as np

np.set_printoptions(precision=3)

# Przygotowujemy parametry przetwarzania
datasets = {
    'Euro28_A': ('E28A', 81, 'Euro'),
    'Euro28_G': ('E28G', 81, 'Euro'),
    'US26_A': ('U26A', 83, 'US'),
    'US26_G': ('U26G', 83, 'US')
}

# Iterujemy zbiory danych
for filename in datasets:
    # Load dataset
    dbname, f, kao = datasets[filename] # kao - tid

    print(dbname)

    awf_X_kao_candidates = np.load("results/%s-awf-X.npy" % filename)
    awf_y_kao_candidates = np.load("results/%s-awf-y.npy" % filename).astype(int)

    bwf_X_kao_candidates = np.load("results/%s-bwf-X.npy" % filename)
    bwf_y_kao_candidates = np.load("results/%s-bwf-y.npy" % filename).astype(int)

    a_X_kao_candidates = np.load("results/%s-a-X.npy" % filename)
    a_y_kao_candidates = np.load("results/%s-a-y.npy" % filename).astype(int)

    r_X_kao_candidates = np.load("results/%s-r-X.npy" % filename)
    r_y_kao_candidates = np.load("results/%s-r-y.npy" % filename).astype(int)

    o_X_kao_candidates = np.load("results/%s-o-X.npy" % filename)
    o_y_kao_candidates = np.load("results/%s-o-y.npy" % filename).astype(int)

    print("AWF")
    awf = np.concatenate((awf_X_kao_candidates.T, [awf_y_kao_candidates])).T
    print(awf)

    print("BWF")
    bwf = np.concatenate((bwf_X_kao_candidates.T, [bwf_y_kao_candidates])).T
    print(bwf)

    print("A")
    a = np.concatenate((a_X_kao_candidates.T, [a_y_kao_candidates])).T
    print(a)

    print("R")
    r = np.concatenate((r_X_kao_candidates.T, [r_y_kao_candidates])).T
    print(r)

    print("O")
    o = np.concatenate((o_X_kao_candidates.T, [o_y_kao_candidates])).T
    print(o)

    np.savetxt('results/%s-awf.csv' % filename, awf, delimiter=',', fmt="%i")
    np.savetxt('results/%s-bwf.csv' % filename, bwf, delimiter=',', fmt="%i")
    np.savetxt('results/%s-a.csv' % filename, a, delimiter=',', fmt="%i")
    np.savetxt('results/%s-r.csv' % filename, r, delimiter=',', fmt="%i")
    np.savetxt('results/%s-o.csv' % filename, o, delimiter=',', fmt="%i")
