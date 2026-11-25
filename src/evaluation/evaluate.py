
import numpy as np
import time

from src.evaluation.metrics import recall_at_k


def evaluate_method(
    name,
    search_fn,
    queries,
    truth,
    k=10,
    repeats=3
):
    queries = np.asarray(queries)
    truth = np.asarray(truth)

    if queries.ndim != 2:
        raise ValueError("queries phải là mảng 2 chiều")

    if truth.ndim != 2:
        raise ValueError("truth phải là mảng 2 chiều")

    if len(queries) == 0:
        raise ValueError("Không có truy vấn để đánh giá")

    if len(queries) != len(truth):
        raise ValueError("Số lượng queries và truth không khớp")

    if k < 1:
        raise ValueError("k phải lớn hơn hoặc bằng 1")

    if repeats < 1:
        raise ValueError("repeats phải lớn hơn hoặc bằng 1")

    runs = []
    result = None

    for _ in range(repeats):
        start = time.perf_counter()

        output = search_fn(queries, k)

        elapsed = time.perf_counter() - start

        # Một số thuật toán trả về (IDs, scores)
        if isinstance(output, tuple):
            output = output[0]

        result = np.asarray(output)

        if result.ndim != 2:
            raise ValueError("Kết quả tìm kiếm phải là mảng 2 chiều")

        if result.shape[0] != len(queries):
            raise ValueError("Số lượng kết quả không khớp số truy vấn")

        runs.append(elapsed * 1000 / len(queries))

    latency = float(np.median(runs))

    recall = recall_at_k(result, truth, k)

    qps = 1000 / latency if latency > 0 else np.nan

    metrics = {
        "method": name,
        f"recall@{k}": recall,
        "ms_per_query": latency,
        "queries_per_second": qps
    }

    return metrics, result
