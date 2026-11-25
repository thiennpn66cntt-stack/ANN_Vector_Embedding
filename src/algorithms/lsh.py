import numpy as np


class RandomHyperplaneLSH:

    def __init__(self, n_bits=12, n_tables=4, seed=42):
        self.n_bits = n_bits
        self.n_tables = n_tables
        self.seed = seed
        self.planes = []
        self.tables = []
        self.codes = []
        self.X = None

    def _hash_codes(self, X, planes):
        signs = (X @ planes.T) >= 0
        weights = 1 << np.arange(self.n_bits, dtype=np.uint64)
        return (signs.astype(np.uint64) * weights).sum(axis=1)

    def fit(self, X):
        self.X = np.asarray(X, dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        self.planes = []
        self.tables = []
        self.codes = []
        d = self.X.shape[1]

        for _ in range(self.n_tables):
            planes = rng.normal(size=(self.n_bits, d)).astype(np.float32)
            codes = self._hash_codes(self.X, planes)
            table = {}

            for idx, code_value in enumerate(codes):
                table.setdefault(int(code_value), []).append(idx)

            self.planes.append(planes)
            self.tables.append(table)
            self.codes.append(codes)

        return self

    def _candidate_ids(self, q, min_candidates=100):
        candidates = set()

        for planes, table, codes in zip(self.planes, self.tables, self.codes):
            code = int(self._hash_codes(q[None, :], planes)[0])
            candidates.update(table.get(code, []))

            if len(candidates) < min_candidates:
                for bit in range(self.n_bits):
                    neighbor_code = code ^ (1 << bit)
                    candidates.update(table.get(neighbor_code, []))

                    if len(candidates) >= min_candidates:
                        break

        if candidates:
            return np.fromiter(candidates, dtype=np.int64)

        return np.array([], dtype=np.int64)

    def search_one(self, q, k=10, min_candidates=100):
        if self.X is None:
            raise ValueError("Phải gọi fit(X) trước khi tìm kiếm.")

        if k < 1 or k > len(self.X):
            raise ValueError("k phải nằm trong khoảng từ 1 đến số vector dữ liệu.")

        if min_candidates < 1:
            raise ValueError("min_candidates phải lớn hơn 0.")

        q = np.asarray(q, dtype=np.float32)

        cand = self._candidate_ids(q, max(min_candidates, k))

        if len(cand) < k:
            cand = np.arange(len(self.X), dtype=np.int64)

        scores = self.X[cand] @ q
        top = np.argsort(-scores)[:k]

        return cand[top]

    def search(self, Q, k=10, min_candidates=100):
        if self.X is None:
            raise ValueError("Phải gọi fit(X) trước khi tìm kiếm.")

        Q = np.asarray(Q, dtype=np.float32)

        results = [
            self.search_one(q, k, min_candidates)
            for q in Q
        ]

        return np.stack(results).astype(np.int64)
