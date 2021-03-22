import numpy as np
from sklearn.utils import resample

kaos = ['Euro', 'US']

ratio = .1
for kao in kaos:
    print(kao)
    X_kao = np.load('kangurki/%s.npy' % kao)

    n_samples = int(X_kao.shape[0]*ratio)

    small_X_kao = resample(X_kao, n_samples=n_samples,
                           random_state=1410)

    print(X_kao.shape, small_X_kao.shape)
    np.save("kangurki/small_%s" % kao, small_X_kao)
