"""Elliptic Bitcoin data set (Kaggle: ellipticco/elliptic-data-set) -> numpy arrays used by cbdc_rev.

usage: python scripts/prep_data.py /path/to/elliptic_bitcoin_dataset  data/

Reads elliptic_txs_features.csv, elliptic_txs_classes.csv, elliptic_txs_edgelist.csv and writes X.npy (N x 165, the time-step column is excluded),
ts.npy (time step 1..49), y.npy (1 illicit, 0 licit, -1 unknown), txid.npy, src.npy, dst.npy (edge list as row indices).
The data set itself is NOT redistributed with this repository.
"""
import sys, os, time
import numpy as np, pandas as pd

raw, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
t0 = time.time()
feats = pd.read_csv(f"{raw}/elliptic_txs_features.csv", header=None)
classes = pd.read_csv(f"{raw}/elliptic_txs_classes.csv")
edges = pd.read_csv(f"{raw}/elliptic_txs_edgelist.csv")
txid = feats.iloc[:, 0].values.astype(np.int64)
ts = feats.iloc[:, 1].values.astype(np.int64)
X = feats.iloc[:, 2:].values.astype(np.float32)
cls = classes.set_index("txId")["class"].reindex(txid).astype(str).values
y = np.where(cls == "1", 1, np.where(cls == "2", 0, -1)).astype(np.int8)
idx = {t: i for i, t in enumerate(txid)}
src = edges["txId1"].map(idx).values.astype(np.int32)
dst = edges["txId2"].map(idx).values.astype(np.int32)
print("nodes", len(txid), "features", X.shape[1], "edges", len(src), "| illicit", int((y == 1).sum()), "licit", int((y == 0).sum()), "unknown", int((y == -1).sum()))
print("edges between different time steps:", int((ts[src] != ts[dst]).sum()))
for name, lo, hi in [("train", 1, 28), ("val", 29, 34), ("test", 35, 49), ("test 35-42", 35, 42), ("test 43-49", 43, 49)]:
    m = (ts >= lo) & (ts <= hi) & (y >= 0)
    print(f"{name:11s} labelled={m.sum():6d} illicit={(y[m] == 1).sum():5d} ({100 * (y[m] == 1).mean():.1f} %)")
np.save(f"{out}/X.npy", X); np.save(f"{out}/ts.npy", ts); np.save(f"{out}/y.npy", y)
np.save(f"{out}/txid.npy", txid); np.save(f"{out}/src.npy", src); np.save(f"{out}/dst.npy", dst)
print("saved in %.1f s" % (time.time() - t0))
