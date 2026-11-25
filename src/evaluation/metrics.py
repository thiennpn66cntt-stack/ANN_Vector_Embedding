
import numpy as np


def recall_at_k(result_ids, truth_ids, k=10):
    result_ids = np.asarray(result_ids)
    truth_ids = np.asarray(truth_ids)

    if k < 1:
        raise ValueError("k phải lớn hơn hoặc bằng 1")

    if result_ids.ndim != 2 or truth_ids.ndim != 2:
        raise ValueError("result_ids và truth_ids phải là mảng 2 chiều")

    if len(result_ids) != len(truth_ids):
        raise ValueError("Số lượng kết quả và ground truth không khớp")

    if len(result_ids) == 0:
        raise ValueError("Không có truy vấn để đánh giá")

    recalls = []

    for pred, truth in zip(result_ids, truth_ids):
        pred_set = set(map(int, pred[:k]))
        truth_set = set(map(int, truth[:k]))

        if len(truth_set) == 0:
            raise ValueError("Ground truth không được rỗng")

        correct = len(pred_set & truth_set)
        recall = correct / len(truth_set)

        recalls.append(recall)

    return float(np.mean(recalls))
