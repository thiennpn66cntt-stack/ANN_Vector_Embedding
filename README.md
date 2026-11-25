# ANN Vector Embedding – Tìm kiếm vector ngữ nghĩa tiếng Việt

## 1. Giới thiệu dự án

ANN Vector Embedding là dự án nghiên cứu và triển khai các phương pháp tìm kiếm láng giềng gần đúng (*Approximate Nearest Neighbor – ANN*) trên dữ liệu văn bản tiếng Việt đã được biểu diễn dưới dạng vector embedding.

Dự án tập trung vào việc:

- Chuẩn bị và kiểm tra dữ liệu văn bản tiếng Việt.
- Biểu diễn văn bản thành vector embedding.
- Xây dựng phương pháp tìm kiếm chính xác làm mốc đối chiếu.
- Cài đặt và đánh giá các thuật toán tìm kiếm gần đúng.
- So sánh độ chính xác và tốc độ tìm kiếm bằng các chỉ số Recall@10, thời gian truy vấn và QPS.
- Tách mã nguồn từ notebook thành các module Python trong thư mục `src/`.
- Viết unit test để kiểm tra các thuật toán và chức năng đánh giá.

Toàn bộ quá trình phát triển, thực nghiệm và lưu trữ dữ liệu của dự án được thực hiện trên Google Colab kết hợp Google Drive.

## 2. Môi trường và công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ lập trình | Python |
| Môi trường thực hiện | Google Colab |
| Lưu trữ dự án | Google Drive |
| Xử lý mảng số | NumPy |
| Xử lý dữ liệu bảng | Pandas |
| Phân cụm dữ liệu | Scikit-learn – MiniBatchKMeans |
| Tạo vector embedding | Sentence Transformers |
| Mô hình embedding | `intfloat/multilingual-e5-small` |
| Tăng tốc tính toán trong Brute Force | PyTorch |
| Thanh tiến trình | tqdm |
| Kiểm thử | pytest |
| Chỉ số đánh giá | Recall@K, Latency, QPS |

### 2.1. Cài đặt thư viện

Trong môi trường Python/Google Colab, có thể cài đặt các thư viện được sử dụng bằng lệnh:

```bash
pip install numpy pandas scikit-learn sentence-transformers torch tqdm pytest
```

Notebook còn sử dụng FAISS trong phần thực nghiệm đối chiếu HNSW và IVF của thư viện. Nếu chạy lại các ô thực nghiệm FAISS, cần cài đặt gói FAISS phù hợp với môi trường Colab:

```bash
pip install faiss-cpu
```

Lưu ý: các module tự cài trong `src/` và bộ unit test không cần chạy toàn bộ phần thực nghiệm FAISS.

## 3. Thông tin dữ liệu

### 3.1. Dữ liệu đầu vào

Theo cấu hình và kết quả kiểm tra dữ liệu trong notebook:

| Thành phần | Giá trị |
|---|---:|
| Ngôn ngữ corpus | Tiếng Việt |
| Số câu corpus ban đầu | 20.000 |
| Số truy vấn | 500 |
| Số chiều embedding | 384 |
| Mô hình embedding | `intfloat/multilingual-e5-small` |
| Số kết quả cần tìm | Top-10 |
| Ground Truth | Top-10 được tạo bằng Brute Force |

### 3.2. Các tệp dữ liệu chính

| Tệp | Nội dung |
|---|---|
| `corpus_vi.txt` | Văn bản corpus tiếng Việt |
| `queries_vi.txt` | Văn bản truy vấn tiếng Việt |
| `corpus.csv` | Corpus ở dạng CSV |
| `queries.csv` | Truy vấn ở dạng CSV |
| `corpus_embeddings.npy` | Vector embedding của corpus |
| `query_embeddings.npy` | Vector embedding của truy vấn |
| `corpus_clean.csv` | Corpus sau bước lọc dữ liệu |
| `corpus_embeddings_clean.npy` | Embedding tương ứng với corpus sau lọc |

Các vector được lưu bằng định dạng NumPy `.npy`. Trong notebook, corpus embedding có kích thước `(20000, 384)` và query embedding có kích thước `(500, 384)`.

### 3.3. Ground Truth

Ground Truth được tạo bằng phương pháp Brute Force. Với mỗi truy vấn, thuật toán tính độ tương đồng với toàn bộ vector trong corpus và lấy ra 10 kết quả tốt nhất.

Các tệp Ground Truth:

- `results/ground_truth_top10.npy`: ID của 10 kết quả chính xác cho mỗi truy vấn.
- `results/ground_truth_scores.npy`: điểm tương đồng tương ứng.

Ground Truth được sử dụng làm dữ liệu chuẩn để tính Recall@10 cho các phương pháp tìm kiếm gần đúng.

## 4. Quy trình kiểm tra và làm sạch dữ liệu

### 4.1. Kiểm tra dữ liệu

Notebook thực hiện kiểm tra một số dấu hiệu dữ liệu cần xem xét:

1. Câu có ít hơn 8 từ.
2. Câu bị trùng lặp sau khi chuyển về chữ thường.
3. Câu có tỷ lệ ký tự đặc biệt lớn hơn 30%.
4. Câu có hai từ liên tiếp giống nhau.

Kết quả kiểm tra corpus được ghi nhận trong notebook:

| Tiêu chí | Số câu được phát hiện |
|---|---:|
| Câu dưới 8 từ | 0 |
| Câu trùng lặp | 0 |
| Câu có tỷ lệ ký tự đặc biệt trên 30% | 2 |
| Câu có hai từ liên tiếp giống nhau | 197 |

Các tiêu chí ký tự đặc biệt và từ lặp liên tiếp là bước phát hiện câu cần xem xét. Chúng không đồng nghĩa rằng tất cả câu được phát hiện đều là câu rác. Ví dụ, cụm từ lặp có thể hợp lệ trong tiếng Việt.

### 4.2. Module làm sạch `src/data/clean_data.py`

Module cung cấp hai hàm:

**`clean_sentence(sentence, min_words=8)`**

- Kiểm tra đầu vào có phải chuỗi hay không.
- Chuẩn hóa các khoảng trắng liên tiếp thành một khoảng trắng.
- Loại bỏ khoảng trắng ở đầu và cuối câu.
- Loại bỏ chuỗi rỗng.
- Loại bỏ chuỗi không có ký tự chữ.
- Loại bỏ câu có số từ nhỏ hơn `min_words`.
- Trả về câu đã chuẩn hóa nếu hợp lệ; ngược lại trả về `None`.

**`clean_corpus(texts, min_words=8)`**

- Duyệt qua danh sách câu.
- Gọi `clean_sentence` để kiểm tra từng câu.
- Bỏ qua câu không hợp lệ.
- Dùng `casefold()` để chuẩn hóa khi so sánh trùng lặp.
- Chỉ giữ lại một câu cho mỗi nội dung trùng nhau.
- Trả về danh sách câu đã làm sạch.

Trong lần làm sạch corpus chính được ghi nhận tại notebook (cell 74), quy trình còn kết hợp thêm danh sách `REMOVE_IDS` và `NOISE_IDS` để loại các câu đã xác định có lỗi hoặc là tiêu đề, danh sách, bảng và nội dung tham khảo. Sau đó, chương trình dùng biểu thức chính quy để loại nội dung có dấu hiệu mã bảng Wikipedia (`colspan` hoặc `rowspan`), đồng thời loại câu trùng lặp theo `casefold()`. Vì vậy, kết quả làm sạch thực tế không chỉ dựa riêng vào hai hàm được mô tả ở trên.

### 4.3. Giữ đúng quan hệ giữa câu và embedding

Khi lọc corpus, notebook kiểm tra số lượng câu trong `corpus.csv` bằng số vector trong `corpus_embeddings.npy` trước khi xử lý.

Trong quá trình duyệt từng dòng, nếu câu được giữ lại thì embedding tại cùng chỉ số dòng cũng được giữ lại. Cách làm này bảo đảm câu văn bản và vector embedding tương ứng không bị lệch nhau. Mỗi dòng dữ liệu sạch còn lưu `clean_index` mới, `corpus_id` gốc và nội dung câu.

Hai tệp đầu ra được lưu:

- `data/corpus_clean.csv`
- `data/corpus_embeddings_clean.npy`

**Kết quả thực tế được ghi nhận tại cell 74–76 của notebook:**

| Thành phần | Trước khi lọc | Sau khi lọc |
|---|---:|---:|
| Số câu | 20.000 | 19.947 |
| Kích thước embedding | (20000, 384) | (19947, 384) |

Tổng cộng có 53 câu bị loại, gồm 52 câu thuộc danh sách ID được xác định trước (`REMOVE_IDS` sau khi hợp nhất với `NOISE_IDS`) và 1 câu bị loại bởi bộ lọc tự động. Câu bị loại tự động có `corpus_id` 6862, chứa nội dung dạng mã bảng Wikipedia (`rowspan=2`).

Notebook kiểm tra và xác nhận số câu khớp với số vector embedding, `corpus_id` không trùng lặp, `clean_index` liên tục, đồng thời embedding không chứa giá trị không hợp lệ. Số câu bị thay đổi nội dung trong quá trình làm sạch là 0.

Lưu ý: các tiêu chí tỷ lệ ký tự đặc biệt và từ lặp liên tiếp được thống kê để kiểm tra; chúng không được dùng trực tiếp làm điều kiện loại tự động trong quy trình tại cell 74.

## 5. Các thuật toán tìm kiếm được triển khai

### 5.1. Brute Force

Brute Force là phương pháp tìm kiếm chính xác. Với mỗi vector truy vấn, thuật toán tính độ tương đồng giữa truy vấn và toàn bộ vector trong database, sau đó chọn Top-K kết quả có điểm cao nhất.

Trong module `src/algorithms/brute_force.py`, hàm `exact_search` sử dụng phép nhân ma trận để tính điểm tương đồng theo từng batch. Module sử dụng PyTorch và có thể chuyển tensor sang CUDA nếu môi trường hỗ trợ.

Đặc điểm:

- So sánh truy vấn với toàn bộ database.
- Trả về ID và điểm tương đồng của Top-K.
- Là phương pháp đối chiếu độ chính xác.
- Được sử dụng để tạo Ground Truth.

### 5.2. LSH – Locality Sensitive Hashing

LSH trong dự án sử dụng phương pháp Random Hyperplane.

Ý tưởng chính:

- Sử dụng các siêu phẳng ngẫu nhiên để tạo mã nhị phân cho vector.
- Chia dữ liệu vào các bảng băm.
- Khi truy vấn, tìm các vector có mã băm phù hợp trong các bảng.
- Tính độ tương đồng trên tập ứng viên và trả về Top-K.

Các tham số được khảo sát trong notebook gồm số bit và số bảng băm (`n_bits`, `n_tables`). Việc thay đổi các tham số này giúp quan sát ảnh hưởng đến Recall@10 và tốc độ truy vấn.

Module: `src/algorithms/lsh.py`.

### 5.3. IVF – Inverted File Index

IVF chia database thành các cụm vector bằng MiniBatchKMeans.

Quy trình tổng quát:

1. Phân cụm vector thành `nlist` cụm.
2. Xác định các cụm gần với vector truy vấn.
3. Chỉ tìm kiếm trong một số cụm được chọn.
4. Tính độ tương đồng và lấy Top-K kết quả.

Tham số `nprobe` điều khiển số cụm được khảo sát khi tìm kiếm. Tăng số cụm được khảo sát thường giúp mở rộng tập ứng viên, nhưng cũng có thể làm tăng thời gian truy vấn.

Module: `src/algorithms/ivf.py`.

### 5.4. NSW – Navigable Small World

NSW là thuật toán tìm kiếm dựa trên đồ thị láng giềng. Mỗi vector tương ứng với một đỉnh; các cạnh thể hiện quan hệ láng giềng giữa các vector.

Module `src/algorithms/nsw.py` triển khai NSW một tầng với các bước chính sau:

**Xây dựng đồ thị (`fit(X)`)**

1. Chuyển dữ liệu đầu vào thành mảng NumPy kiểu `float32`.
2. Khởi tạo danh sách kề cho các đỉnh.
3. Đưa đỉnh đầu tiên (ID 0) vào đồ thị.
4. Chèn lần lượt các đỉnh từ ID 1 đến ID `n-1`.
5. Với mỗi đỉnh mới, gọi `_search_layer` để tìm ứng viên trong phần đồ thị đã được xây dựng.
6. Tham số tìm kiếm khi xây dựng là `max(M, build_candidates)`.
7. Chọn tối đa `M` ứng viên tốt nhất làm láng giềng cho đỉnh mới.
8. Thêm liên kết ngược từ các láng giềng về đỉnh mới.
9. Gọi `_prune_neighbors` để giới hạn số láng giềng của đỉnh được cập nhật.

**Tìm kiếm trên đồ thị (`_search_layer`)**

- Sử dụng heap để quản lý danh sách ứng viên cần mở rộng.
- Sử dụng một danh sách kết quả có giới hạn bởi `ef`.
- Theo dõi các đỉnh đã thăm để tránh xử lý lặp.
- Mở rộng các láng giềng chưa thăm.
- Giữ lại tối đa `ef` kết quả tốt nhất trong quá trình tìm kiếm.

**Tìm kiếm Top-K (`search_one`, `search`)**

- Bắt đầu từ đỉnh 0.
- Tìm kiếm trên đồ thị bằng danh sách ứng viên `ef`.
- Sắp xếp ứng viên theo độ tương đồng với truy vấn.
- Trả về tối đa `k` ID.
- Hàm `search` thực hiện tìm kiếm cho nhiều truy vấn.

Các tham số chính:

| Tham số | Ý nghĩa |
|---|---|
| `M` | Số láng giềng tối đa được giữ cho mỗi đỉnh sau bước cắt tỉa |
| `build_candidates` | Quy mô ứng viên dùng khi chèn đỉnh; giá trị `ef` xây dựng là `max(M, build_candidates)` |
| `ef` | Giới hạn danh sách kết quả/ứng viên trong tìm kiếm trên đồ thị |
| `k` | Số kết quả cần trả về |

Trong lần kiểm tra NSW được lưu trong notebook, mô hình được khởi tạo với `M=12`, `build_candidates=100`, tìm kiếm 5 truy vấn với `k=10`, `ef=100`. Kết quả có kích thước `(5, 10)` và đồ thị có 20.000 đỉnh.

## 6. Đánh giá thuật toán

### 6.1. Recall@K

Recall@K đo tỷ lệ kết quả đúng trong Top-K mà thuật toán tìm kiếm trả về so với Top-K Ground Truth.

Trong dự án, Recall@10 được dùng làm chỉ số chính để so sánh chất lượng tìm kiếm.

- Recall@10 càng gần 1 thì kết quả tìm kiếm càng gần với Ground Truth.
- Brute Force được dùng làm phương pháp tìm kiếm chính xác để tạo Ground Truth.
- Các phương pháp ANN được đánh giá bằng cách so sánh ID dự đoán với ID Ground Truth.

### 6.2. Latency

`ms_per_query` là thời gian trung bình cho một truy vấn, tính bằng mili giây theo cách đo trong chương trình đánh giá.

Latency càng thấp thì thời gian xử lý mỗi truy vấn càng nhanh.

### 6.3. QPS

QPS (*Queries Per Second*) biểu thị số truy vấn xử lý được trong một giây. Trong module đánh giá:

`QPS = 1000 / ms_per_query`

QPS càng cao thì thông lượng truy vấn càng lớn.

### 6.4. Hàm đánh giá

Module `src/evaluation/metrics.py` chứa hàm `recall_at_k`.

Module `src/evaluation/evaluate.py` chứa hàm `evaluate_method`, thực hiện:

- Kiểm tra định dạng và số lượng queries, truth.
- Kiểm tra `k` và số lần lặp.
- Đo thời gian thực thi hàm tìm kiếm.
- Tính latency theo mỗi truy vấn.
- Tính Recall@K.
- Tính QPS.
- Trả về dictionary chỉ số và ID kết quả.

## 7. Cấu trúc thư mục dự án

Cấu trúc dưới đây mô tả các thành phần chính được tổ chức trong dự án:

```text
ANN_Vector_Embedding/
│
├── data/
│   ├── corpus_vi.txt
│   ├── queries_vi.txt
│   ├── corpus.csv
│   ├── queries.csv
│   ├── corpus_embeddings.npy
│   ├── query_embeddings.npy
│   ├── corpus_clean.csv
│   ├── corpus_embeddings_clean.npy
│   └── processed/
│       └── dataset_clean.csv
│
├── results/
│   ├── ground_truth_top10.npy
│   ├── ground_truth_scores.npy
│   ├── comparison_results.csv
│   ├── experiment_summary.json
│   ├── recall_vs_qps.png
│   ├── recall_vs_latency.png
│   ├── lsh_*_top10.npy
│   ├── ivf_*_top10.npy
│   ├── nsw_*_top10.npy
│   └── faiss_*_top10.npy
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── clean_data.py
│   │
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── brute_force.py
│   │   ├── lsh.py
│   │   ├── ivf.py
│   │   └── nsw.py
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py
│       └── evaluate.py
│
├── tests/
│   ├── __init__.py
│   └── test_algorithms.py
│
└── README.md
```

Một số tệp kết quả có thể thay đổi theo lần chạy thực nghiệm. Cấu trúc trên liệt kê các nhóm tệp chính, không yêu cầu mọi tệp kết quả phải luôn tồn tại ngay sau khi cài đặt.

## 8. Mô tả các module trong `src/`

| Đường dẫn | Chức năng |
|---|---|
| `src/data/clean_data.py` | Chuẩn hóa, kiểm tra câu và loại câu không hợp lệ/trùng lặp |
| `src/algorithms/brute_force.py` | Tìm kiếm chính xác bằng Brute Force |
| `src/algorithms/lsh.py` | Cài đặt Random Hyperplane LSH |
| `src/algorithms/ivf.py` | Cài đặt tìm kiếm IVF |
| `src/algorithms/nsw.py` | Cài đặt đồ thị NSW một tầng |
| `src/evaluation/metrics.py` | Tính Recall@K |
| `src/evaluation/evaluate.py` | Đánh giá Recall, Latency và QPS |

Các module được tách riêng để thuận tiện cho việc import, kiểm thử và mở rộng mà không phải chạy lại toàn bộ notebook.

## 9. Hướng dẫn chạy dự án

### 9.1. Mở dự án trên Google Colab

1. Mở Google Colab.
2. Kết nối Google Drive.
3. Đặt thư mục dự án tại:

```text
/content/drive/MyDrive/ANN_Vector_Embedding
```

4. Chuyển thư mục làm việc đến thư mục dự án:

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/ANN_Vector_Embedding
```

### 9.2. Kiểm tra import các module

Chạy từ thư mục gốc dự án:

```python
import sys

PROJECT_PATH = "/content/drive/MyDrive/ANN_Vector_Embedding"

if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

from src.algorithms.brute_force import exact_search
from src.algorithms.lsh import RandomHyperplaneLSH
from src.algorithms.ivf import SimpleIVF
from src.algorithms.nsw import SimpleNSW

from src.data.clean_data import clean_sentence, clean_corpus
from src.evaluation.metrics import recall_at_k
from src.evaluation.evaluate import evaluate_method

print("Import các module thành công.")
```

### 9.3. Chạy unit test

Đảm bảo đang ở thư mục gốc `ANN_Vector_Embedding`, sau đó chạy:

```bash
python -m pytest tests/test_algorithms.py -v
```

Chạy nhanh và chỉ hiển thị tổng kết:

```bash
python -m pytest tests/test_algorithms.py -q
```

Có thể chạy riêng một nhóm test bằng `-k`, ví dụ:

```bash
python -m pytest tests/test_algorithms.py -k nsw -v
```

### 9.4. Lưu ý khi chạy lại thực nghiệm

- Cần có các tệp dữ liệu và embedding trong thư mục `data/`.
- Cần có Ground Truth trong thư mục `results/` nếu muốn tính Recall@10.
- Nên kiểm tra số dòng corpus bằng số vector embedding trước khi huấn luyện/tìm kiếm.
- Khi sử dụng corpus đã lọc, phải sử dụng embedding tương ứng với corpus đó và tạo Ground Truth tương ứng; không trộn ID hoặc embedding giữa dữ liệu gốc và dữ liệu đã lọc.
- Các kết quả thời gian phụ thuộc vào phần cứng, phiên Colab và cấu hình chạy.

## 10. Unit test

Bộ kiểm thử nằm tại:

```text
tests/test_algorithms.py
```

Bộ test sử dụng dữ liệu giả lập có seed cố định để kết quả kiểm thử có thể tái lập. Dữ liệu kiểm thử gồm 100 vector, mỗi vector có 32 chiều và 5 vector truy vấn lấy từ database.

### 10.1. Danh sách 7 unit test

| STT | Tên test | Nội dung kiểm tra |
|---:|---|---|
| 1 | `test_brute_force` | Kích thước ID/score, ID hợp lệ, tìm được chính vector truy vấn và điểm được sắp xếp giảm dần |
| 2 | `test_lsh` | Khởi tạo, xây dựng bảng băm, kích thước kết quả, ID hợp lệ và không trùng trong từng truy vấn |
| 3 | `test_ivf` | Phân cụm bằng MiniBatchKMeans, tìm kiếm, kích thước kết quả và ID hợp lệ |
| 4 | `test_nsw` | Xây dựng đồ thị NSW, chèn tuần tự, tìm kiếm bằng `ef`, kích thước kết quả và ID hợp lệ |
| 5 | `test_recall_at_k` | Kiểm tra giá trị Recall@5 với dữ liệu dự đoán và Ground Truth mẫu |
| 6 | `test_recall_invalid_input` | Kiểm tra trường hợp số lượng truy vấn và Ground Truth không khớp phải phát sinh `ValueError` |
| 7 | `test_evaluate_method` | Kiểm tra hàm đánh giá, Recall, latency, QPS và kích thước kết quả |

### 10.2. Kết quả kiểm thử trong notebook

Notebook ghi nhận lần chạy đầy đủ:

```text
tests/test_algorithms.py::test_brute_force PASSED
tests/test_algorithms.py::test_lsh PASSED
tests/test_algorithms.py::test_ivf PASSED
tests/test_algorithms.py::test_nsw PASSED
tests/test_algorithms.py::test_recall_at_k PASSED
tests/test_algorithms.py::test_recall_invalid_input PASSED
tests/test_algorithms.py::test_evaluate_method PASSED

7 passed in 7.36s
```

Kết quả trên là đầu ra của lần chạy được lưu trong notebook. Khi nộp hoặc chạy lại trên môi trường khác, nên chạy pytest lại và cập nhật thời gian thực tế nếu cần.

## 11. Kết quả thực nghiệm được ghi nhận

Notebook lưu kết quả so sánh trong `results/comparison_results.csv`, gồm 38 cấu hình/phương pháp và các chỉ số như:

- `method`
- `recall@10`
- `ms_per_query`
- `queries_per_second`
- Các tham số tương ứng như `n_bits`, `n_tables`, `nlist`, `nprobe`, `M`, `efSearch`.

### 11.1. Một số kết quả NSW

Các kết quả thực nghiệm và bảng so sánh trong mục 11 được ghi nhận trên corpus gốc gồm 20.000 vector, với 500 truy vấn và Ground Truth Top-10 tương ứng được tạo từ corpus gốc. Đây là bộ dữ liệu dùng cho các tệp kết quả thực nghiệm đã lưu trong `results/`, không phải kết quả đánh giá lại trên corpus sạch gồm 19.947 vector.

Theo bảng kết quả được in trong notebook:

| Phương pháp | Recall@10 | ms/query | QPS |
|---|---:|---:|---:|
| NSW M=8 | 0.0174 | 0.072638 | 13.766,91 |
| NSW M=12 | 0.0294 | 0.116731 | 8.566,72 |
| NSW M=20 | 0.0460 | 0.111227 | 8.990,66 |

Các số liệu trên là kết quả của lần thực nghiệm đã lưu. Chúng cho thấy trong cấu hình được thử nghiệm, NSW tự cài có tốc độ truy vấn cao nhưng Recall@10 còn thấp. Vì vậy, cần tiếp tục cải thiện chất lượng đồ thị và/hoặc tham số tìm kiếm nếu mục tiêu là tăng độ chính xác.

### 11.2. Một số kết quả đối chiếu trong bảng tổng hợp

| Phương pháp | Recall@10 | ms/query | QPS |
|---|---:|---:|---:|
| Brute Force | 1.0000 | 0.796324 | 1.255,77 |
| FAISS HNSW M=32, efSearch=256 | 0.9990 | 1.287810 | 776,51 |
| FAISS HNSW M=16, efSearch=256 | 0.9988 | 0.795702 | 1.256,75 |
| FAISS HNSW M=32, efSearch=128 | 0.9962 | 0.729833 | 1.370,18 |
| FAISS IVF nprobe=50 | 0.9960 | 1.497788 | 667,65 |

Các phương pháp FAISS HNSW và FAISS IVF là phần đối chiếu sử dụng thư viện FAISS trong notebook; chúng không phải module tự cài trong `src/`.

### 11.3. Tệp kết quả

Các tệp kết quả chính được notebook kiểm tra gồm:

- `comparison_results.csv`: bảng so sánh các cấu hình.
- `experiment_summary.json`: thông tin tổng hợp thực nghiệm.
- `*_top10.npy`: ID Top-10 của từng phương pháp/cấu hình.
- `recall_vs_qps.png`: biểu đồ Recall@10 theo QPS.
- `recall_vs_latency.png`: biểu đồ Recall@10 theo Latency.

## 12. Giới hạn và hướng phát triển

Qua kết quả thực nghiệm được ghi nhận:

1. Chất lượng Recall@10 của NSW tự cài còn thấp so với các cấu hình FAISS HNSW trong bảng đối chiếu.
2. Kết quả tìm kiếm phụ thuộc vào các tham số như `M`, `build_candidates` và `ef`.
3. Các quy tắc làm sạch hiện tại chủ yếu xử lý chuỗi rỗng, chuỗi không có chữ, câu ngắn và câu trùng lặp; các câu có tỷ lệ ký tự đặc biệt cao hoặc từ lặp liên tiếp mới được thống kê để xem xét.
4. Khi thay đổi corpus, cần bảo đảm embedding, ID và Ground Truth được đồng bộ.
5. Thời gian truy vấn và QPS có thể thay đổi theo phần cứng và môi trường thực thi.

Hướng phát triển:

- Thử nghiệm thêm các cấu hình `M`, `build_candidates` và `ef` cho NSW.
- Đánh giá Recall@10 trên toàn bộ tập truy vấn sau mỗi thay đổi.
- Bổ sung unit test cho các trường hợp dữ liệu biên của hàm làm sạch và NSW.
- Hoàn thiện quy trình tạo lại embedding và Ground Truth khi corpus thay đổi.
- Tiếp tục so sánh độ chính xác và tốc độ giữa các thuật toán.

## 13. Kết luận

Dự án đã tổ chức mã nguồn tìm kiếm vector thành các module Python riêng trong `src/`, bao gồm Brute Force, LSH, IVF, NSW, xử lý dữ liệu và đánh giá. Bộ kiểm thử `pytest` gồm 7 test đã có lần chạy ghi nhận thành công. Notebook cũng lưu các kết quả thực nghiệm, bảng so sánh và biểu đồ đánh giá.

Quy trình làm sạch dữ liệu đã loại 53 câu khỏi corpus ban đầu 20.000 câu, còn lại 19.947 câu cùng 19.947 vector embedding tương ứng. Các kết quả thực nghiệm đã lưu trong README vẫn thuộc lần đánh giá trên corpus gốc 20.000 vector; nếu chuyển sang đánh giá corpus sạch thì cần tạo lại Ground Truth và kết quả tìm kiếm tương ứng.

README này mô tả cấu trúc, cách sử dụng, quy trình kiểm thử và các kết quả đã được ghi nhận trong notebook, làm tài liệu hướng dẫn cho việc tiếp tục phát triển và đánh giá dự án.
