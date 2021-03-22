import numpy as np
import pandas as pd
import config

for filename in config.datasets:
    dbname, f = config.datasets[filename]
    print(filename, dbname, f)

    df = pd.read_csv("datasets/%s.txt" % filename, sep="\t")

    # Get X, y
    X = df.values[:, 6:7 + f]
    y = df['AT'].values

    # Prepare column for category
    r = df[df['ID'].str.contains("r")].index.values
    a = df[df['ID'].str.contains("a")].index.values
    b = df[df['ID'].str.contains("b")].index.values
    c = np.zeros(y.shape).astype(int)
    c[r] = 0
    c[a] = 1
    c[b] = 2

    # Create dataset
    ds = np.concatenate((X, c[:, np.newaxis], y[:, np.newaxis]), axis=1)
    np.save("datasets/%s" % dbname, ds)
