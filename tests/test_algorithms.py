
import sys
from pathlib import Path

import numpy as np
import pytest

# Xác định thư mục gốc dự án
PROJECT_PATH = Path(__file__).resolve().parents[1]

if str(PROJECT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_PATH))

# Import các thuật toán
from src.algorithms.brute_force import exact_search
from src.algorithms.lsh import RandomHyperplaneLSH
from src.algorithms.ivf import SimpleIVF
from src.algorithms.nsw import SimpleNSW

# Import các hàm đánh giá
from src.evaluation.metrics import recall_at_k
from src.evaluation.evaluate import evaluate_method


# ============================================================
# HÀM TẠO DỮ LIỆU KIỂM THỬ
# ============================================================

def create_test_data():
    """
    Tạo dữ liệu giả lập để kiểm thử thuật toán.

    Database:
        100 vector, mỗi vector có 32 chiều.

    Queries:
        5 vector lấy từ database.

    Sử dụng seed cố định để kết quả có thể tái lập.
    """

    rng = np.random.default_rng(42)

    X = rng.normal(size=(100, 32)).astype(np.float32)

    # Chuẩn hóa vector
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    Q = X[:5].copy()

    return X, Q


# ============================================================
# TEST 1: KIỂM TRA BRUTE FORCE
# ============================================================

def test_brute_force():
    """
    Kiểm tra:
    - Thuật toán chạy thành công.
    - Kết quả có đúng kích thước.
    - ID nằm trong phạm vi database.
    - Truy vấn trùng với vector trong database
      phải tìm được chính vector đó ở vị trí đầu tiên.
    """

    X, Q = create_test_data()

    ids, scores = exact_search(
        Q,
        X,
        k=10
    )

    assert ids.shape == (5, 10)

    assert scores.shape == (5, 10)

    assert np.all(ids >= 0)

    assert np.all(ids < len(X))

    # Kiểm tra vector truy vấn được tìm thấy chính nó
    for i in range(len(Q)):
        assert ids[i, 0] == i

    # Điểm tương đồng được sắp xếp giảm dần
    assert np.all(
        scores[:, :-1] >= scores[:, 1:] - 1e-6
    )


# ============================================================
# TEST 2: KIỂM TRA LSH
# ============================================================

def test_lsh():
    """
    Kiểm tra:
    - Khởi tạo LSH.
    - Xây dựng bảng băm.
    - Tìm kiếm Top-10.
    - Kết quả có đúng kích thước.
    - ID hợp lệ và không bị lặp trong từng truy vấn.
    """

    X, Q = create_test_data()

    lsh = RandomHyperplaneLSH(
        n_bits=8,
        n_tables=4,
        seed=42
    )

    lsh.fit(X)

    result = lsh.search(
        Q,
        k=10,
        min_candidates=10
    )

    assert result.shape == (5, 10)

    assert np.all(result >= 0)

    assert np.all(result < len(X))

    for row in result:
        assert len(set(row.tolist())) == len(row)


# ============================================================
# TEST 3: KIỂM TRA IVF
# ============================================================

def test_ivf():
    """
    Kiểm tra:
    - Phân cụm dữ liệu bằng MiniBatchKMeans.
    - Tìm kiếm trên các cụm gần nhất.
    - Kết quả có đúng kích thước.
    - ID nằm trong phạm vi database.

    nprobe=5 tương ứng kiểm tra cả 5 cụm.
    """

    X, Q = create_test_data()

    ivf = SimpleIVF(
        nlist=5,
        random_state=42
    )

    ivf.fit(X)

    result = ivf.search(
        Q,
        k=10,
        nprobe=5
    )

    assert result.shape == (5, 10)

    assert np.all(result >= 0)

    assert np.all(result < len(X))


# ============================================================
# TEST 4: KIỂM TRA NSW
# ============================================================

def test_nsw():
    """
    Kiểm tra:
    - Xây dựng đồ thị NSW.
    - Chèn vector tuần tự.
    - Tìm kiếm bằng danh sách ứng viên ef.
    - Kết quả có đúng kích thước.
    - ID nằm trong phạm vi database.
    """

    X, Q = create_test_data()

    nsw = SimpleNSW(
        M=8,
        build_candidates=20,
        seed=42
    )

    nsw.fit(X)

    result = nsw.search(
        Q,
        k=10,
        ef=30
    )

    assert result.shape == (5, 10)

    assert np.all(result >= 0)

    assert np.all(result < len(X))


# ============================================================
# TEST 5: KIỂM TRA RECALL@K
# ============================================================

def test_recall_at_k():
    """
    Kiểm tra độ chính xác của hàm Recall@K.

    Kết quả mong đợi:
    Query 1: 3/5
    Query 2: 2/5

    Recall trung bình = 0.5
    """

    truth = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10]
    ])

    pred = np.array([
        [1, 2, 3, 8, 9],
        [6, 7, 11, 12, 13]
    ])

    recall = recall_at_k(
        pred,
        truth,
        k=5
    )

    assert abs(recall - 0.5) < 1e-6


# ============================================================
# TEST 6: KIỂM TRA XỬ LÝ DỮ LIỆU KHÔNG HỢP LỆ
# ============================================================

def test_recall_invalid_input():
    """
    Kiểm tra hàm Recall@K phát hiện
    số lượng truy vấn không khớp.
    """

    pred = np.array([
        [1, 2, 3]
    ])

    truth = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    with pytest.raises(ValueError):
        recall_at_k(
            pred,
            truth,
            k=3
        )


# ============================================================
# TEST 7: KIỂM TRA EVALUATE METHOD
# ============================================================

def test_evaluate_method():
    """
    Kiểm tra:
    - Hàm đánh giá thuật toán.
    - Tính Recall@K.
    - Tính thời gian truy vấn.
    - Tính Queries Per Second.
    """

    queries = np.eye(5)

    truth = np.tile(
        np.arange(5),
        (5, 1)
    )

    def dummy_search(Q, k):
        return np.tile(
            np.arange(k),
            (len(Q), 1)
        )

    metrics, result = evaluate_method(
        name="Dummy Search",
        search_fn=dummy_search,
        queries=queries,
        truth=truth,
        k=5,
        repeats=3
    )

    assert result.shape == (5, 5)

    assert metrics["method"] == "Dummy Search"

    assert abs(metrics["recall@5"] - 1.0) < 1e-6

    assert metrics["ms_per_query"] >= 0

    assert metrics["queries_per_second"] >= 0
