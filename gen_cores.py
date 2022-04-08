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

    approaches = {
        'awf': np.genfromtxt('results/%s-awf.csv' % filename, delimiter=',').astype(int)[:,:-1],
        'bwf': np.genfromtxt('results/%s-bwf.csv' % filename, delimiter=',').astype(int)[:,:-1],
        'a': np.genfromtxt('results/%s-a.csv' % filename, delimiter=',').astype(int)[:,:-1],
        'r': np.genfromtxt('results/%s-r.csv' % filename, delimiter=',').astype(int)[:,:-1],
        'o': np.genfromtxt('results/%s-o.csv' % filename, delimiter=',').astype(int)[:,:-1]
    }

    for approach in approaches:
        solutions = approaches[approach]

        print(approach)

        for i, cores in enumerate(solutions):
            print(cores)
            filename = "%s_%s_%i" % (
                dbname, approach, i
            )
            print(filename)
            np.savetxt('foo.txt', cores, fmt='%i')
            np.savetxt('cores/%s.cores' % filename, cores, fmt='%i')
