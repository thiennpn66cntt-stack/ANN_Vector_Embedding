
import heapq
import numpy as np
from tqdm.auto import tqdm


class SimpleNSW:

    def __init__(self, M=12, build_candidates=100, seed=42):
        self.M = M
        self.build_candidates = build_candidates
        self.seed = seed
        self.X = None
        self.graph = None

    def _similarity(self, q, idx):
        return float(np.dot(self.X[idx], q))

    def _search_layer(self, q, entry_points, ef, max_id):
        """
        Tìm kiếm trên đồ thị bằng danh sách ứng viên ef.
        Chỉ tìm trong các đỉnh có ID nhỏ hơn max_id.
        """

        visited = set()
        candidates = []
        results = []

        for ep in entry_points:
            if ep < 0 or ep >= max_id or ep in visited:
                continue

            score = self._similarity(q, ep)

            visited.add(ep)
            heapq.heappush(candidates, (-score, ep))
            heapq.heappush(results, (score, ep))

        while candidates:

            current_score, current = heapq.heappop(candidates)
            current_score = -current_score

            if len(results) >= ef:
                if current_score < results[0][0]:
                    break

            for neighbor in self.graph[current]:

                if neighbor >= max_id or neighbor in visited:
                    continue

                visited.add(neighbor)

                score = self._similarity(q, neighbor)

                if len(results) < ef or score > results[0][0]:

                    heapq.heappush(
                        candidates,
                        (-score, neighbor)
                    )

                    heapq.heappush(
                        results,
                        (score, neighbor)
                    )

                    if len(results) > ef:
                        heapq.heappop(results)

        results.sort(key=lambda x: x[0], reverse=True)

        return [idx for score, idx in results]

    def _prune_neighbors(self, node):
        """
        Giữ lại tối đa M láng giềng gần nhất.
        """

        neighbors = list(dict.fromkeys(self.graph[node]))

        if not neighbors:
            return

        scores = self.X[neighbors] @ self.X[node]

        take = min(self.M, len(neighbors))

        best = np.argpartition(-scores, take - 1)[:take]

        self.graph[node] = [
            neighbors[i] for i in best
        ]

    def fit(self, X):
        """
        Xây dựng đồ thị NSW bằng cách chèn lần lượt từng vector.
        """

        self.X = np.asarray(X, dtype=np.float32)

        n, d = self.X.shape

        if n == 0:
            raise ValueError("Dữ liệu X không được rỗng.")

        if self.M < 1:
            raise ValueError("M phải lớn hơn hoặc bằng 1.")

        self.graph = [[] for _ in range(n)]

        # Đỉnh đầu tiên được đưa vào đồ thị.
        self.graph[0] = []

        # Chèn lần lượt các đỉnh từ 1 đến n-1.
        for i in tqdm(
            range(1, n),
            desc=f"Build NSW M={self.M}"
        ):

            # Tìm các ứng viên trong đồ thị đã có.
            candidates = self._search_layer(
                q=self.X[i],
                entry_points=[0],
                ef=max(self.M, self.build_candidates),
                max_id=i
            )

            # Chọn tối đa M láng giềng tốt nhất.
            neighbors = candidates[:self.M]

            # Kết nối đỉnh mới với các láng giềng.
            self.graph[i] = list(neighbors)

            for neighbor in neighbors:

                self.graph[neighbor].append(i)

                # Giới hạn số cạnh của đỉnh láng giềng.
                self._prune_neighbors(neighbor)

        return self

    def search_one(self, q, k=10, ef=100, max_steps=None):
        """
        Tìm kiếm Top-K bằng danh sách ứng viên ef.
        """

        if self.X is None or self.graph is None:
            raise ValueError("Phải gọi fit(X) trước khi tìm kiếm.")

        if k < 1:
            raise ValueError("k phải lớn hơn 0.")

        if ef < k:
            raise ValueError("ef phải lớn hơn hoặc bằng k.")

        if len(self.X) == 0:
            raise ValueError("Đồ thị không có dữ liệu.")

        q = np.asarray(q, dtype=np.float32)

        # Điểm bắt đầu tìm kiếm là đỉnh 0.
        candidates = self._search_layer(
            q=q,
            entry_points=[0],
            ef=ef,
            max_id=len(self.X)
        )

        # Tính lại độ tương đồng và sắp xếp.
        scores = self.X[candidates] @ q

        order = np.argsort(-scores)

        result = np.asarray(candidates)[order[:k]]

        # Giữ định dạng đầu ra có đủ k phần tử.
        if len(result) < k:
            result = np.pad(
                result,
                (0, k - len(result)),
                constant_values=0
            )

        return result.astype(np.int64)

    def search(self, Q, k=10, ef=100):
        """
        Tìm kiếm Top-K cho nhiều truy vấn.
        """

        Q = np.asarray(Q, dtype=np.float32)

        return np.asarray([
            self.search_one(q, k=k, ef=ef)
            for q in Q
        ], dtype=np.int64)
