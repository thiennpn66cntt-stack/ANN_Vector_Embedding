import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm.auto import tqdm


SEED = 42


class SimpleIVF:
    def __init__(self, nlist=100, random_state=42):
        self.nlist = nlist
        self.kmeans = None
        self.inverted_lists = None
        self.X = None

    def fit(self, X):
        self.X = X
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.nlist, random_state=SEED,
            batch_size=2048, n_init=3, max_iter=100
        )
        labels = self.kmeans.fit_predict(X)
        self.inverted_lists = [np.where(labels == i)[0] for i in range(self.nlist)]
        return self

    def search(self, Q, k=10, nprobe=5):
        centers = self.kmeans.cluster_centers_.astype('float32')
        out = []
        for q in Q:
            center_scores = centers @ q
            probes = np.argpartition(-center_scores, nprobe-1)[:nprobe]
            cand = np.concatenate([self.inverted_lists[i] for i in probes])
            if len(cand) == 0:
                out.append(np.zeros(k, dtype=np.int64))
                continue
            scores = self.X[cand] @ q
            take = min(k, len(cand))
            local = np.argpartition(-scores, take-1)[:take]
            local = local[np.argsort(-scores[local])]
            chosen = cand[local]
            if len(chosen) < k:
                # Bổ sung nếu cụm quá nhỏ.
                chosen = np.pad(chosen, (0, k-len(chosen)), constant_values=chosen[-1])
            out.append(chosen[:k])
        return np.asarray(out, dtype=np.int64)
