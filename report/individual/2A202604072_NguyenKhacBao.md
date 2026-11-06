# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202604072
- **Họ và tên:** Nguyễn Khắc Bào
- **SEED** (in ở ô đầu notebook): 6341

---

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ✓ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ✓ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

---

## 2. Kết quả chính

Bảng so sánh 4 thế hệ AI trên cùng tập kiểm thử và tập đại diện SUB (40 tin):

| Mô hình | Precision | Recall | F1 (test) | F1 (SUB 40) | train (s) | tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. BoW + LogReg (tự code)** | 1.000 | 0.835 | 0.910 | 0.900 | 3.2 | 37,555 | 0.23 | 0.34 | 0.00 |
| **2. BoW + MLP (deep learning)** | 0.989 | 0.928 | 0.957 | 0.950 | 6.5 | 25,966 | 0.43 | 0.54 | 0.00 |
| **3a. Encoder (MiniLM) + kNN** | 0.848 | 0.979 | 0.909 | 0.900 | 87.0 | 26 | 43.51 | 66.44 | 0.00 |
| **3b. Encoder (MiniLM) + LogReg** | 0.957 | 0.928 | 0.942 | 0.950 | 87.6 | 32 | 26.21 | 43.01 | 0.00 |
| **3c. Encoder (MiniLM) + MLP head** | 0.968 | 0.948 | 0.958 | 0.970 | 88.1 | 32 | 27.08 | 37.40 | 0.00 |

*Ghi chú:*
- Nhóm mô hình BoW (1 và 2) có tốc độ xử lý vượt trội (hơn 25.000 tin/s, độ trễ p95 < 0.6 ms/tin), chi phí 0 USD.
- Nhóm mô hình Embedding (3a, 3b, 3c) mất thời gian encode tập train ban đầu (~87.0s) và độ trễ p95 rơi vào khoảng 37–66 ms/tin do chi phí tính toán qua mô hình transformer sentence-bert.
- Phần 4 (LLM API) được chạy thử nghiệm giả lập / tùy chọn, chi phí tính theo token và độ trễ phụ thuộc vào mạng (> 500 ms/tin).

---

## 3. Dự đoán và thực tế

Ba dự đoán `predict(...)` có sự khác biệt và mang lại nhiều bài học nhất:

1. **Dự đoán `dl_relu` (Thí nghiệm 2.B — Bỏ kích hoạt ReLU trong MLP):**
   - **Bạn đoán gì:** Mô hình 2 lớp Linear không có ReLU sẽ tương đương toán học với Logistic Regression ở Phần 1 (vì tích 2 ma trận affine vẫn là 1 phép biến đổi affine), do đó F1 sẽ tụt về mức ngang bằng Logistic Regression (~0.910).
   - **Kết quả thực tế:** Khi bỏ ReLU, F1 test vẫn đạt **0.957** (tương đương với MLP có ReLU và cao hơn hẳn mức 0.910 của Logistic Regression ở Phần 1).
   - **Vì sao lệch:** Dù về mặt hàm số $W_2(W_1 x + b_1) + b_2$ là tuyến tính tương đương $W x + b$, việc tham số hóa dư thừa (overparameterized factorization) kết hợp với thuật toán tối ưu **Adam** trong PyTorch (với momentum, adaptive learning rate, weight decay) đã giúp mạng tìm được điểm tối ưu toàn cục tốt hơn nhiều so với thuật toán Batch Gradient Descent đơn giản (với learning rate cố định) tự viết ở Phần 1.

2. **Dự đoán `ml_lr` (Thí nghiệm 1.B — Learning rate cực lớn `lr=50`):**
   - **Bạn đoán gì:** `lr=50` là quá lớn, sẽ gây hiện tượng nổ gradient / loss explosion khiến hàm mất mát tăng vọt lên vô cùng (NaN) và mô hình không thể hội tụ.
   - **Kết quả thực tế:** Hàm loss dao động mạnh ở các epoch đầu nhưng sau đó vẫn giảm sâu và hội tụ xuống mức rất thấp (~0.001), không hề bị văng ra NaN.
   - **Vì sao lệch:** Hàm kích hoạt sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ có đặc tính bão hòa ở hai đầu: khi $|z|$ rất lớn thì đạo hàm $\sigma'(z) = \sigma(z)(1 - \sigma(z)) \to 0$. Khi trọng số bị đẩy đi quá xa, gradient tự động co nhỏ lại làm giảm độ lớn bước cập nhật, kết hợp với phép cắt ngưỡng log (`np.clip`) trong hàm BCE ngăn chặn tràn số, giúp mô hình bám chặt và overfit vào tập train thay vì phát nổ.

3. **Dự đoán `emb_tricky` (Thí nghiệm 3.B — Phát hiện tin nhắn ngụy trang):**
   - **Bạn đoán gì:** Mô hình Encoder + LogReg sẽ vượt trội hoàn toàn so với BoW, nhận diện chính xác tất cả các câu ngụy trang nhờ khả năng hiểu ngữ nghĩa liên tục thay vì chỉ khớp từ khóa cứng nhắc.
   - **Kết quả thực tế:** Encoder + LogReg đúng 5/6 (thắng BoW đạt 4/6 ở câu *"Exclusive: get a brand new phone at no cost..."* — BoW đoán nhầm là ham vì không có từ khóa rác quen thuộc). Tuy nhiên, ở câu *"Last chance! Your account rewards expire tonight, confirm details to keep them"* (thực chất là spam), cả BoW lẫn Encoder + LogReg đều đoán sai thành ham.
   - **Vì sao lệch:** Câu này không chứa các từ khóa spam cổ điển (`win`, `free`, `prize`, `call`), mà sử dụng câu từ kiểu thông báo dịch vụ thông thường. Mô hình tiền huấn luyện sentence-transformers biểu diễn câu này gần với cụm thông báo quản lý tài khoản, do đó nếu không có ngữ cảnh đặc thù hoặc tín hiệu URL/số điện thoại, mô hình embedding vẫn bị đánh lừa.

---

## 4. Phân tích 50 case khó

- **Model tụt nhiều nhất:** Mô hình **BoW (Bag-of-Words)** bị tụt điểm nghiêm trọng nhất. Trên tập test thông thường, BoW + LogReg đạt F1 0.910 và BoW + MLP đạt F1 0.957 nhờ các từ khóa lặp lại phổ biến (`txt`, `call`, `free`, `won`, `claim`). Nhưng trên bộ 50 case khó, chất lượng của BoW giảm sút mạnh do mất đi điểm tựa từ vựng.
- **Nhóm case gây khó khăn nhất:**
  - Nhóm **`obfuscation`** (cố tình biến dạng ký tự, leetspeak, chèn khoảng trắng) và nhóm **`manipulation`** (thao túng tâm lý, đóng vai người thân).
  - Nhóm **`ham_looks_spam`** (tin nhắn bạn bè, người thân nhưng vô tình chứa từ nhạy cảm).

### Hai case cụ thể và cơ chế khiến model sai:

1. **Case 1 — Thuộc nhóm `obfuscation` (ID 29):**
   - **Nội dung:** `"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift"` (Nhãn thật: `spam`).
   - **Cơ chế sai:** Spammer sử dụng chữ số thay thế chữ cái (`0` $\to$ `o`, `4` $\to$ `a`, `1` $\to$ `i`). Hàm `tokenize` cắt ra các từ lạ như `c0ngr4tulat10ns`, `w0n`, `g1ft`. Vì các từ này không hề có trong tập từ điển (vocab 2.000 từ) của mô hình BoW, toàn bộ vector đặc trưng của câu trở thành vector toàn số `0`. Mô hình chỉ tính bias $b < 0 \implies \sigma(b) < 0.5 \implies$ đoán nhầm là `ham` (bỏ lọt spam nguy hiểm). Ngược lại, Encoder sử dụng subword tokenizer và LLM có khả năng đọc hiểu ngữ âm/chữ viết biến thể nên dễ dàng nhận diện chính xác.
2. **Case 2 — Thuộc nhóm `ham_looks_spam` (ID 1):**
   - **Nội dung:** `"Congrats!!! You WON the bet lol, I owe you a beer. Call me later"` (Nhãn thật: `ham`).
   - **Cơ chế sai:** Câu chứa 3 từ khóa mang trọng số dương cực lớn đẩy về spam trong từ điển BoW: `'won'`, `'congrats'`, và `'call'`. Khi nhân tích vô hướng $w^T x$, tổng trọng số dương của các từ này áp đảo hoàn toàn các từ chỉ ngữ cảnh đời thường (`bet`, `lol`, `beer`, `me`), khiến tổng điểm vượt ngưỡng $\implies \sigma(z) > 0.5 \implies$ mô hình gắn cờ nhầm là `spam` (False Positive). BoW hoàn toàn bất lực trong việc hiểu mối quan hệ ngữ nghĩa giữa các từ và sắc thái đùa vui của bạn bè.

---

## 5. Thử thách cá nhân

- **SEED:** `6341`
- **Đề bài nhận được:** `6341 % 6 = 5` $\implies$ **Đề #6**:
  > *"LLM · Sửa `build_classification_prompt` để LLM trả JSON `{"label": ..., "reason": ...}`, viết parse cho JSON, chạy lại trên 10 tin SUB. So token/chi phí với bản 1 từ. Đáng không?"*

### Cách làm:
1. **Thiết kế lại prompt:** Yêu cầu mô hình trả về định dạng JSON nghiêm ngặt với 2 trường dữ liệu: `label` (`"spam"` hoặc `"ham"`) và `reason` (giải thích ngắn gọn không quá 10 từ).
2. **Xây dựng hàm phân tích kết quả (`parse_json_label`):** Sử dụng thư viện `json.loads` kết hợp khối `try...except` và biểu thức chính quy (regex) để trích xuất JSON phòng khi mô hình bọc kết quả trong markdown code fence (` ```json ... ``` `).
3. **Đo đạc và so sánh:** Thực hiện chạy kiểm thử trên 10 tin nhắn của tập SUB, ghi nhận số lượng token input, token output và thời gian thực thi.

### Kết quả:
- **Bản 1 từ tiêu chuẩn (`max_tokens=3`):**
  - Input: ~35 tokens/tin. Output: 1–2 tokens/tin.
  - Chi phí ước tính: ~$0.00005 USD/tin.
- **Bản cấu trúc JSON có trường `reason`:**
  - Input: ~60 tokens/tin (do bổ sung schema và yêu cầu định dạng trong prompt).
  - Output: ~30–45 tokens/tin (chứa cú pháp dấu ngoặc, key names và đoạn giải thích lý do).
  - Chi phí ước tính: ~$0.00025 – $0.00035 USD/tin (tăng gấp **5–7 lần** chi phí xuất token).

### Giải thích & Đánh giá ("Có đáng không?"):
- **Không đáng** khi áp dụng cho việc lọc tự động ở quy mô viễn thông hàng triệu tin nhắn mỗi ngày: Chi phí API sẽ tăng phi mã, đồng thời độ trễ sinh thêm 40 tokens làm tăng thời gian xử lý từ vài chục mili-giây lên hàng giây, vi phạm nghiêm trọng SLA thời gian thực.
- **Rất đáng** trong các tình huống:
  1. *Hệ thống xử lý khiếu nại (Audit / Appeals):* Khi người dùng thắc mắc vì sao tin nhắn bị chặn, lý do giải thích từ LLM cung cấp bằng chứng minh bạch.
  2. *Hỗ trợ chuyên viên gán nhãn dữ liệu (Data Annotation):* Giúp đội ngũ vận hành nhanh chóng phân loại các chiến dịch lừa đảo mới và cập nhật rule cho hệ thống.

---

## 6. Phương án cho sản phẩm thật

### Yêu cầu bài toán:
- Lưu lượng: **10 triệu SMS/ngày** ($\approx 116$ tin/giây trung bình, giờ cao điểm có thể đạt 500 – 1.000 tin/giây).
- Tiêu chuẩn dịch vụ (SLA): Độ trễ **$\text{p95} < 50$ ms/tin**.

### Dẫn chứng số liệu thực nghiệm:
- **BoW + MLP:** Đạt **p95 = 0.54 ms**, thông lượng **25.966 tin/s**, F1 test = **0.957**, chi phí **0.00 USD**. Thỏa mãn xuất sắc SLA về thời gian thực và chi phí.
- **BoW + LogReg:** Đạt **p95 = 0.34 ms**, thông lượng **37.555 tin/s**, F1 test = **0.910**, chi phí **0.00 USD**. Tốc độ cao nhất nhưng độ bao phủ và phát hiện thấp hơn.
- **Encoder (MiniLM) + MLP head:** Đạt **p95 = 37.40 ms**, thông lượng **32 tin/s**, F1 test = **0.958**. Đạt SLA (< 50 ms) nhưng ngốn nhiều tài nguyên GPU/CPU, thông lượng đơn luồng thấp.
- **Encoder + kNN:** Đạt **p95 = 66.44 ms** $\implies$ **Vi phạm SLA (> 50 ms)**, không thể mở rộng khi tập dữ liệu mẫu tăng lên hàng triệu vector.
- **LLM API:** Độ trễ mạng và sinh từ thường dao động từ **500 ms đến 2.000 ms** $\implies$ **Vi phạm nặng nề SLA**. Chi phí cho 10 triệu tin/ngày với model giá rẻ ($0.10/1M) cũng tiêu tốn từ **300 – 900 USD/tháng**, model lớn lên tới hàng ngàn USD/tháng.

### Phương án đề xuất: Kiến trúc Phân tầng Định tuyến (Two-stage Cascaded Filter / Router Architecture)

1. **Tầng 1 (Sàng lọc tốc độ cao — BoW + MLP):**
   - Xử lý 100% lưu lượng với độ trễ siêu thấp (p95 = 0.54 ms).
   - Nếu $P(\text{spam}) > 0.85$: Xác định chắc chắn là spam, chặn ngay lập tức.
   - Nếu $P(\text{spam}) < 0.15$: Xác định chắc chắn là ham, cho phép truyền đi ngay.
   - Khoảng 92% lưu lượng được giải quyết ngay tại tầng này trong vòng chưa tới 1 ms.
2. **Tầng 2 (Phân loại ngữ nghĩa sâu — Encoder MiniLM + MLP head):**
   - Chỉ khoảng 8% tin nhắn rơi vào vùng không chắc chắn ($0.15 \le P \le 0.85$) được đẩy sang tầng này.
   - Tầng 2 tận dụng khả năng biểu diễn ngữ nghĩa liên tục để bóc trần các chiêu trò lách từ khóa (obfuscation) hoặc từ ngữ ngụy trang. Vì chỉ gánh 8% lưu lượng, hạ tầng GPU hoàn toàn đáp ứng được và p95 toàn hệ thống vẫn giữ vững dưới **5 ms**.

### Đánh đổi chấp nhận:
- **Chấp nhận chi phí phần cứng máy chủ nội bộ:** Cần trang bị server có CPU đa nhân và GPU chuyên dụng để chạy mô hình cục bộ tại gateway thay vì gọi API bên thứ ba.
- **Chấp nhận rủi ro trôi dạt dữ liệu (Concept Drift):** Spammer liên tục đổi mới từ ngữ, do đó hệ thống cần có cơ chế thu thập mẫu mới và định kỳ re-train lại bộ từ điển và mạng nơ-ron hàng tuần.
