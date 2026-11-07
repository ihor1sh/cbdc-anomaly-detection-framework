"""Shared data access for the revision experiments (Article 2.1).

Protocol fixed here (the one the manuscript states):
  train = time steps 1-28, validation = 29-34, test = 35-49 (labelled nodes only).
Graph models are trained on the sub-graph of steps <= 34 and evaluated on the
disjoint sub-graph of steps >= 35 (Elliptic has no edges between time steps), i.e.
the test graph and its nodes are never seen during training (inductive evaluation).
"""
import os
import numpy as np

ROOT = os.environ.get("CBDC_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DATA_DIR = os.environ.get("CBDC_DATA", os.path.join(ROOT, "data"))
SEEDS = [42, 1337, 2024, 7, 31415]
TRAIN_END, VAL_END, TEST_END = 28, 34, 49
KYC_COLS = ["country_risk", "kyc_tier", "kyc_age_days", "pep_flag", "aml_history"]


class Elliptic:
    """Raw Elliptic arrays (165 features, time step excluded) and fixed masks."""

    def __init__(self, data_dir=DATA_DIR):
        self.X = np.load(f"{data_dir}/X.npy")            # (N, 165) float32, globally z-scored by the dataset authors
        self.ts = np.load(f"{data_dir}/ts.npy")           # time step 1..49
        self.y = np.load(f"{data_dir}/y.npy")             # 1 illicit, 0 licit, -1 unknown
        self.src = np.load(f"{data_dir}/src.npy").astype(np.int64)
        self.dst = np.load(f"{data_dir}/dst.npy").astype(np.int64)
        self.n = len(self.y)
        lab = self.y >= 0
        self.m_train = lab & (self.ts <= TRAIN_END)
        self.m_val = lab & (self.ts > TRAIN_END) & (self.ts <= VAL_END)
        self.m_test = lab & (self.ts > VAL_END)
        self.in_trg = self.ts <= VAL_END                  # nodes of the training graph (train + val steps)
        self.in_teg = ~self.in_trg                        # nodes of the (disjoint) test graph

    def subgraph(self, node_mask):
        """Return (nodes, local_src, local_dst) for the induced sub-graph (undirected edges listed once)."""
        nodes = np.where(node_mask)[0]
        remap = -np.ones(self.n, dtype=np.int64)
        remap[nodes] = np.arange(len(nodes))
        keep = node_mask[self.src] & node_mask[self.dst]
        return nodes, remap[self.src[keep]], remap[self.dst[keep]]


def class_weights(y_train):
    pos = int((y_train == 1).sum()); neg = int((y_train == 0).sum())
    return neg, pos, max(1.0, neg / max(1, pos))
