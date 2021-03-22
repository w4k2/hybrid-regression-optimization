"""
Weryfikacja jakości modelu regresji w predykcji rozwiazan z optymalizatorow.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import r2_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.base import clone
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm

np.set_printoptions(precision=3)

# Definicja metryki odciętej
def cevs(y_true, y_pred):
    return np.clip(explained_variance_score(y_true, y_pred), 0, 1)

# Przygotowujemy parametry przetwarzania
n_splits = 5
n_best_kaos = 10
R, O = (0, 1)
mlp_params = {
    "solver": "lbfgs",
    "hidden_layer_sizes": (50),
    "random_state": 216 + 80082
}
datasets = {
    'Euro28_A': ('E28A', 81, 'Euro'),
    'Euro28_G': ('E28G', 81, 'Euro'),
    'US26_A': ('U26A', 83, 'US'),
    'US26_G': ('U26G', 83, 'US')
}

# Estymatory bazowe
base_reg = MLPRegressor(**mlp_params)
base_clf = MLPClassifier(**mlp_params)
resampler = RandomOverSampler

# Metryki
metric_reg = cevs
metric_clf = balanced_accuracy_score

# Iterujemy zbiory danych
for filename in datasets:
    # Load dataset
    dbname, f, kao = datasets[filename] # kao - identyfikator kangurka
    ds = np.load("datasets/%s.npy" % dbname, allow_pickle=True)
    X = ds[:,:-2]               # dane
    c = ds[:,-2].astype(int)    # kategorie
    c[c==2] = 1                 # spłaszczamy kategorie [r=0, o=1]
    y = ds[:,-1]                # etykiety
    X_kao = np.load('kangurki/%s.npy' % kao) # kangurki

    # Przygotowujemy walidację krzyżową
    skf = StratifiedKFold(n_splits=n_splits, random_state=1410, shuffle=True)

    # Przygotowujemy miejsce na składowanie zespołów
    a_ensemble = []
    r_ensemble = []
    o_ensemble = []
    c_ensemble = []

    # Miejsce na składowanie wyników
    reg_scores = np.zeros((n_splits, 3, 3))     # F x REG x CAT
    clf_scores = np.zeros(n_splits)             # F
    wf_reg_scores = np.zeros((n_splits, 2, 3))  # F x REG x CAT

    # Pętla walidacji krzyżowej
    for fold, (train, test) in tqdm(enumerate(skf.split(X, c)),
                                    total = n_splits):
        # Inicjalizujemy estymatory
        # a - uczony na wszystkich wzorcach
        # r - uczący na wzorcach losowych
        # o - uczący na wzorcach z optymalizacji
        a_est, r_est, o_est = [clone(base_reg) for _ in range(3)]

        # Uczymy estymatory
        a_est.fit(X[train], y[train])
        r_est.fit(X[train][c[train]==R],
                  y[train][c[train]==R])
        o_est.fit(X[train][c[train]==O],
                  y[train][c[train]==O])

        # Przygotowujemy i uczymy klasyfikator kategorii
        # [balansując resamplerem]
        clf = clone(base_clf).fit(*resampler(random_state=1410)
                                  .fit_resample(X[train], c[train]))

        # Dodajemy je do puli
        a_ensemble.append(a_est)
        r_ensemble.append(r_est)
        o_ensemble.append(o_est)
        c_ensemble.append(clf)

        # Zbieramy wszystkie predykcje i probę z klasyfikacji
        a_y_pred = a_est.predict(X)
        r_y_pred = r_est.predict(X)
        o_y_pred = o_est.predict(X)
        c_y_pred = clf.predict(X)
        c_y_pp = clf.predict_proba(X)

        """
        Do a weird fuck
        """
        # AWF - ważymy regresory R i O probą z klasyfikatora C
        awf_y_pred = r_y_pred * c_y_pp[:,0] + o_y_pred * c_y_pp[:,1]

        # BWF - wynik AWF uśredniamy z regresorem A
        bwf_y_pred = (awf_y_pred + a_y_pred) / 2

        """
        Wyliczanie metryk
        """
        # Wyliczamy wszystkie metryki
        # REGRESJA
        # 0 - kategoria ucząca  [a,r,o]
        # 1 - kategoria testowa [a,r,o]
        a_a_score = metric_reg(y[test], a_y_pred[test])
        r_a_score = metric_reg(y[test], r_y_pred[test])
        o_a_score = metric_reg(y[test], o_y_pred[test])
        a_r_score = metric_reg(y[test][c[test] == R],
                               a_y_pred[test][c[test] == R])
        r_r_score = metric_reg(y[test][c[test] == R],
                               r_y_pred[test][c[test] == R])
        o_r_score = metric_reg(y[test][c[test] == R],
                               o_y_pred[test][c[test] == R])
        a_o_score = metric_reg(y[test][c[test] == O],
                               a_y_pred[test][c[test] == O])
        r_o_score = metric_reg(y[test][c[test] == O],
                               r_y_pred[test][c[test] == O])
        o_o_score = metric_reg(y[test][c[test] == O],
                               o_y_pred[test][c[test] == O])

        # Regresja AWF
        a_a_wf_score = metric_reg(y[test], awf_y_pred[test])
        a_r_wf_score = metric_reg(y[test][c[test] == R],
                                  awf_y_pred[test][c[test] == R])
        a_o_wf_score = metric_reg(y[test][c[test] == O],
                                  awf_y_pred[test][c[test] == O])

        b_a_wf_score = metric_reg(y[test], bwf_y_pred[test])
        b_r_wf_score = metric_reg(y[test][c[test] == R],
                                  bwf_y_pred[test][c[test] == R])
        b_o_wf_score = metric_reg(y[test][c[test] == O],
                                  bwf_y_pred[test][c[test] == O])

        # KLASYFIKACJA
        clf_score = metric_clf(c[test], c_y_pred[test])

        """
        Składowanie i prezentacja wyników
        """
        # Składowanie wyników
        reg_scores[fold] = [
            [a_a_score, a_r_score, a_o_score],
            [r_a_score, r_r_score, r_o_score],
            [o_a_score, o_r_score, o_o_score],
        ]
        wf_reg_scores[fold] = [
            [a_a_wf_score, a_r_wf_score, a_o_wf_score],
            [b_a_wf_score, b_r_wf_score, b_o_wf_score],
        ]
        clf_scores[fold] = clf_score

    print("\nDATASET", filename, dbname, f)
    print("\nBase regressors # [A R O] x [A R O]")
    print(np.mean(reg_scores, axis=0))
    print("\nCategorical classifier")
    print("%.3f (%.2f)" % (np.mean(clf_scores), np.std(clf_scores)))
    print("\nWF regressors # [WFA WFB] x [A R O]")
    print(np.mean(wf_reg_scores, axis=0))

    # Zbieramy informacje z zespołów
    est_a_preds = np.mean(
        np.array([est.predict(X) for est in a_ensemble]), axis=0)
    est_r_preds = np.mean(
        np.array([est.predict(X) for est in r_ensemble]), axis=0)
    est_o_preds = np.mean(
        np.array([est.predict(X) for est in o_ensemble]), axis=0)
    est_c_probas = np.mean(
        np.array([est.predict_proba(X) for est in c_ensemble]), axis=0)
    est_c_preds = np.argmax(est_c_probas, axis=1)

    # AWF - ważymy regresory R i O probą z klasyfikatora C
    awf_y_pred = (
        est_r_preds * est_c_probas[:,0] + est_o_preds * est_c_probas[:,1]
    )

    # BWF - wynik AWF uśredniamy z regresorem A
    bwf_y_pred = (awf_y_pred + est_a_preds) / 2

    # Wyniki ogólne
    print("A %.3f" % metric_reg(y, est_a_preds))
    print("R %.3f" % metric_reg(y[c==R], est_r_preds[c==R]))
    print("O %.3f" % metric_reg(y[c==O], est_o_preds[c==O]))
    print("C %.3f" % metric_clf(c, est_c_preds))

    print("AWF-A %.3f" % metric_reg(y, awf_y_pred))
    print("AWF-R %.3f" % metric_reg(y[c==R], awf_y_pred[c==R]))
    print("AWF-O %.3f" % metric_reg(y[c==O], awf_y_pred[c==O]))

    print("BWF-A %.3f" % metric_reg(y, bwf_y_pred))
    print("BWF-R %.3f" % metric_reg(y[c==R], bwf_y_pred[c==R]))
    print("BWF-O %.3f" % metric_reg(y[c==O], bwf_y_pred[c==O]))

    """
    Kangurze rezultaty
    """
    est_a_preds = np.mean(
        np.array([est.predict(X_kao) for est in a_ensemble]), axis=0)
    est_r_preds = np.mean(
        np.array([est.predict(X_kao) for est in r_ensemble]), axis=0)
    est_o_preds = np.mean(
        np.array([est.predict(X_kao) for est in o_ensemble]), axis=0)
    #est_c_preds = np.array([est.predict(X) for est in c_ensemble])
    est_c_probas = np.mean(
        np.array([est.predict_proba(X_kao) for est in c_ensemble]), axis=0)
    est_c_preds = np.argmax(est_c_probas, axis=1)

    # AWF - ważymy regresory R i O probą z klasyfikatora C
    awf_y_pred = est_r_preds * est_c_probas[:,0] + est_o_preds * est_c_probas[:,1]

    # BWF - wynik AWF uśredniamy z regresorem A
    bwf_y_pred = (awf_y_pred + est_a_preds) / 2

    # A
    a_pred = est_a_preds

    # R
    r_pred = est_r_preds

    # O
    o_pred = est_o_preds

    # Rozkład predykowanych kategorii
    print(np.unique(est_c_preds, return_counts=True)[1]/bwf_y_pred.shape)

    X_kao_candidates = X_kao[est_c_preds==O]
    awf_candidates = awf_y_pred[est_c_preds==O]
    bwf_candidates = bwf_y_pred[est_c_preds==O]
    X_aro_candidates = X_kao

    a_candidates = a_pred
    r_candidates = r_pred
    o_candidates = o_pred

    awf_idx = np.argsort(-awf_candidates)[:n_best_kaos]
    bwf_idx = np.argsort(-bwf_candidates)[:n_best_kaos]
    a_idx = np.argsort(-a_candidates)[:n_best_kaos]
    r_idx = np.argsort(-r_candidates)[:n_best_kaos]
    o_idx = np.argsort(-o_candidates)[:n_best_kaos]

    print("AWF candidates")
    print(awf_idx)
    print(awf_candidates[awf_idx])

    np.save("results/%s-awf-X" % filename, X_kao_candidates[awf_idx])
    np.save("results/%s-awf-y" % filename, awf_candidates[awf_idx])

    print("BWF candidates")
    print(bwf_idx)
    print(bwf_candidates[bwf_idx])

    np.save("results/%s-bwf-X" % filename, X_kao_candidates[bwf_idx])
    np.save("results/%s-bwf-y" % filename, bwf_candidates[bwf_idx])


    print("A candidates")
    print(a_idx)
    print(a_candidates[a_idx])

    np.save("results/%s-a-X" % filename, X_aro_candidates[a_idx])
    np.save("results/%s-a-y" % filename, a_candidates[a_idx])


    print("R candidates")
    print(r_idx)
    print(r_candidates[r_idx])

    np.save("results/%s-r-X" % filename, X_aro_candidates[r_idx])
    np.save("results/%s-r-y" % filename, r_candidates[r_idx])


    print("O candidates")
    print(o_idx)
    print(o_candidates[o_idx])

    np.save("results/%s-o-X" % filename, X_aro_candidates[o_idx])
    np.save("results/%s-o-y" % filename, o_candidates[o_idx])
    # exit()
