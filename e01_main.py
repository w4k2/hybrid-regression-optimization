import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import r2_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.base import clone
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm
from scipy.stats import ttest_rel

np.set_printoptions(precision=3)

# Crop-metric definition
def cevs(y_true, y_pred):
    return np.clip(explained_variance_score(y_true, y_pred), 0, 1)

# Preparing processing parameters
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

# Base estimators
base_reg = MLPRegressor(**mlp_params)
base_clf = MLPClassifier(**mlp_params)
resampler = RandomOverSampler

# Experimental protocol
n_splits = 5
alpha = .05
metric_reg = cevs
metric_clf = balanced_accuracy_score
skf = StratifiedKFold(n_splits=n_splits, random_state=1410,
                      shuffle=True)

"""
Experimental loop
"""
for filename in datasets:
    # Load dataset
    dbname, f, kao = datasets[filename] # kao - tid
    ds = np.load("datasets/%s.npy" % dbname, allow_pickle=True)

    # Establish X, y
    X = ds[:,:-2]                   # patterns
    y_cat = ds[:,-2].astype(int)    # flatten categories
    y_cat[y_cat==2] = 1             # spłaszczamy kategorie [r=0, o=1]
    y_reg = ds[:,-1]                # regression labels
    X_kao = np.load('t_samples/%s.npy' % kao) # t_samples

    print("X", X.shape, "y_cat", y_cat.shape, "y_reg", y_reg.shape,
          "X_kao", X_kao.shape)

    # Storage for scores
    reg_scores = np.zeros((n_splits, 3, 5))     # F x CAT x REG
    clf_scores = np.zeros(n_splits)             # F

    # Prepare lists for ensembles
    a_ensemble = []
    r_ensemble = []
    o_ensemble = []
    c_ensemble = []

    """
    CV loop
    """
    for fold, (train, test) in tqdm(enumerate(skf.split(X, y_cat)),
                                    total = n_splits):
        # Model initialization
        # a - uczony na wszystkich wzorcach
        # r - uczący na wzorcach losowych
        # o - uczący na wzorcach z optymalizacji
        a_reg, r_reg, o_reg = [clone(base_reg) for _ in range(3)]

        # Building models
        a_reg.fit(X[train], y_reg[train])
        r_reg.fit(X[train][y_cat[train]==0], y_reg[train][y_cat[train]==0])
        o_reg.fit(X[train][y_cat[train]!=0], y_reg[train][y_cat[train]!=0])
        clf = clone(base_clf).fit(*resampler(random_state=1410)
                                  .fit_resample(X[train], y_cat[train]))

        # Adding models to pools
        a_ensemble.append(a_reg)
        r_ensemble.append(r_reg)
        o_ensemble.append(o_reg)
        c_ensemble.append(clf)

        # Gather probas and predictions
        a_y_pred = a_reg.predict(X)
        r_y_pred = r_reg.predict(X)
        o_y_pred = o_reg.predict(X)
        c_y_pred = clf.predict(X)
        c_y_pp = clf.predict_proba(X)
        arc_y_pred = np.sum([r_y_pred, o_y_pred] * c_y_pp.T, axis=0)
        aroc_y_pred = (arc_y_pred + a_y_pred) / 2

        # Metric calculations
        ## Prepare
        a_ground = y_cat[test] >= 0
        r_ground = y_cat[test] == 0
        o_ground = y_cat[test] != 0
        grounds = [a_ground, r_ground, o_ground]
        regs_preds = [
            a_y_pred, r_y_pred, o_y_pred,
            arc_y_pred, aroc_y_pred
        ]
        freg_scores = np.zeros((5,3)) # Approach x Category

        ## Calculate
        for pred_idx, y_pred in enumerate(regs_preds):
            for ground_idx, ground in enumerate(grounds):
                score = metric_reg(y_reg[test][ground],
                                   y_pred[test][ground])
                freg_scores[pred_idx, ground_idx] = score
        clf_score = metric_clf(y_cat[test], c_y_pred[test])

        ## Store
        reg_scores[fold] = freg_scores.T
        clf_scores[fold] = clf_score

    print("\nDATASET", filename, dbname, f)

    # Analysis
    mean_reg_scores = np.mean(reg_scores, axis=0)
    std_reg_scores = np.std(reg_scores, axis=0)

    # Statistical analysis
    contexts = ["\\textsc{all}", "\\textsc{ran}", "\\textsc{opt}"]
    for context in range(3):
        relevance = np.array([[ttest_rel(reg_scores[:, context, a_idx],
                                       reg_scores[:, context, b_idx]).pvalue
                             for a_idx in range(5)]
                            for b_idx in range(5)]) < alpha
        superiority = np.array([[ttest_rel(reg_scores[:, context, a_idx],
                                          reg_scores[:, context, b_idx]).statistic
                                for a_idx in range(5)]
                               for b_idx in range(5)]) < 0
        relevant_superiority = relevance * superiority

        conclusion = [list(np.where(_)[0] + 1)
                      for _ in relevant_superiority]

        # LaTeX table
        line_a = mean_reg_scores[context]
        line_b = std_reg_scores[context]
        line_c = conclusion

        line_a = "%s & " % contexts[context] + " & ".join(["%.3f" % v
                                                           for v in line_a])
        line_b = " & " + " & ".join(["\\scriptsize (%.2f)" % v
                                     for v in line_b])
        line_c = " & " + " & ".join([",".join(["\\scriptsize\\oldstylenums{%i}" % vv
                                               for vv in v]) if len(v) > 0 else "---"
                                     for v in line_c])


        print(line_a, "\\\\")
        print(line_b, "\\\\")
        print(line_c, "\\\\")
