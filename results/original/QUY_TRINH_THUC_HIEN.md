# PlanGate: quy trình thực hiện chi tiết

Tài liệu này ghi lại **toàn bộ các bước đã làm**, theo đúng thứ tự thời gian:
- vì sao làm bước đó,
- làm như thế nào (script nào, tham số gì),
- kết quả ra sao,
- kết quả đó dẫn tới quyết định gì ở bước sau.

Phần kết quả để viết paper nằm ở [`ICDE_STORY.md`](ICDE_STORY.md). Tài liệu này tập trung vào **cách làm**, để người khác có thể hiểu, kiểm tra lại và chạy lại.

---

## Mục lục

0. [Bối cảnh và mục tiêu](#0-bối-cảnh-và-mục-tiêu)
1. [Chuẩn bị môi trường](#1-chuẩn-bị-môi-trường)
2. [Giai đoạn 1: prototype với LLM thật (matrix phản thực tế)](#2-giai-đoạn-1-prototype-với-llm-thật)
3. [Giai đoạn 2: phân tích theo từng action (Experiment A, B, C, oracle)](#3-giai-đoạn-2-phân-tích-theo-từng-action)
4. [Giai đoạn 3: planner có ngân sách (Experiment D)](#4-giai-đoạn-3-planner-có-ngân-sách-experiment-d)
5. [Giai đoạn 4: khâu chọn candidate (Experiment J)](#5-giai-đoạn-4-khâu-chọn-candidate-experiment-j)
6. [Giai đoạn 5: nhánh PostgreSQL](#6-giai-đoạn-5-nhánh-postgresql)
7. [Giai đoạn 6: hoàn thiện planner và kiểm định](#7-giai-đoạn-6-hoàn-thiện-planner-và-kiểm-định)
8. [Các quy ước chung (chi phí, latency, thống kê, chống rò rỉ)](#8-các-quy-ước-chung)
9. [Bản đồ file và lệnh chạy lại](#9-bản-đồ-file-và-lệnh-chạy-lại)
10. [Các lỗi đã gặp và cách xử lý](#10-các-lỗi-đã-gặp-và-cách-xử-lý)

---

## 0. Bối cảnh và mục tiêu

**Điểm xuất phát.**
- Repo ban đầu là một scaffold mock: `plangate/planner.py` (planner heuristic), `plangate/runtime.py` (vòng lặp state → action → observation) và 5 câu demo với SQL viết tay.
- Ý tưởng nghiên cứu ban đầu nằm trong `PlanGate_Research_Idea_Summary.md`: thay vì router chọn một lần trước khi sinh SQL, ta lập kế hoạch tuần tự trên cả operator LLM lẫn operator DBMS.

**Mục tiêu của toàn bộ chuỗi experiment** là trả lời ba câu hỏi:
1. Thông tin có được *sau khi sinh SQL*, đặc biệt là bằng chứng từ DB, có giá trị quyết định hơn thông tin chỉ có từ câu hỏi và schema không?
2. Một planner chọn action tuần tự dựa trên bằng chứng có cho đường cost–accuracy tốt hơn các chiến lược chỉ dùng LLM không?
3. Phần nào thực sự thuộc về DBMS, để có một story đủ chất "systems" cho ICDE?

---

## 1. Chuẩn bị môi trường

### 1.1 API LLM

- Dùng MiraiAPI qua client tương thích OpenAI (`base_url=https://api.miraiapi.com/v1`).
- Key `MIRAI_API_KEY` được đặt ở biến môi trường cấp User trên Windows, nhưng tiến trình hiện tại không thừa kế được. Vì vậy `plangate/providers/mirai.py` đọc key theo thứ tự: biến môi trường của tiến trình, rồi registry `HKCU\Environment`.
- Trước khi dùng, mình thử 7 model Trung Quốc bằng một prompt nhỏ (`scratchpad/probe.py`). Cả 7 đều trả lời được: hy3, deepseek-v4-flash, deepseek-v4.1-flash, deepseek-v4-pro, glm-5.3-flash, qwen3.8-max, kimi-k3.
- **Phát hiện quan trọng:** proxy tự chèn thêm khoảng 2–3k prompt token vào mỗi lần gọi. Phần overhead này làm chi phí giữa các model trông gần nhau hơn thực tế, và được ghi lại như một hạn chế.

### 1.2 Thư viện Python

```
pip install openai scikit-learn numpy sqlglot matplotlib "psycopg[binary]"
```

| thư viện | dùng để |
|---|---|
| `sqlglot` | parse SQL, tìm các predicate dạng literal cho value probe |
| `scikit-learn` | các mô hình học (logistic regression, gradient boosting) và cross-validation |
| `psycopg` | kết nối tới PostgreSQL |

### 1.3 Dữ liệu

- **BIRD dev** (346 MB), tải từ `bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip` về `data/bird/dev_20240627/`. Gồm 1534 câu hỏi và 11 database SQLite.
- **BIRD mini-dev** (801 MB). Lần tải đầu bị cụt, phải tải tiếp bằng `curl -C -`. Gồm 500 câu có gold SQL viết cho PostgreSQL, cùng dump PostgreSQL chính thức của cả 11 database.
- **PostgreSQL 16 portable** (bản binary zip của EDB) đặt ở `data/pg/pgsql/`. Máy không có Docker, và gói pip `pgserver` không có wheel cho Python 3.13, nên mới dùng cách này.
- Cả `data/bird/`, `data/pg/` và `cache/` đều đã được thêm vào `.gitignore`.

---

## 2. Giai đoạn 1: prototype với LLM thật

### 2.1 Các module nền

| file | chức năng |
|---|---|
| `plangate/bird.py` | Đọc BIRD. Lấy mẫu phân tầng theo (db_id, độ khó). Mở kết nối SQLite **read-only** (`mode=ro`). Chạy query có timeout bằng progress handler, chỉ cho phép một câu SELECT/WITH duy nhất. Chấm điểm EX: so khớp tập các dòng kết quả, có tính các gold thay thế chính thức trong `dev_tied_append.json`. |
| `plangate/llm_gen.py` | Các template prompt (`direct`, `decompose`, `repair`). Tách SQL từ khối ```sql```. **Cache phản hồi lâu dài:** khóa là hash SHA-256 của (model, messages, temperature, sample id), mỗi khóa một file JSON, có lock riêng cho từng khóa. |
| `plangate/providers/mirai.py` | Client có timeout và retry. Bảng hệ số giá `RATE` theo từng model. |

**Vì sao cache quan trọng.** Mỗi lần gọi LLM chỉ phải trả tiền một lần. Mọi lần phát lại policy sau đó vừa miễn phí vừa tất định. Đây là điều kiện để so sánh công bằng nhiều policy trên **cùng một tập kết quả**.

### 2.2 Pilot: 40 câu × 7 model

Lệnh: `python experiments/run_matrix.py --n 40 --configs pilot`.

**Kết quả:**
- Các model đạt 0.59–0.71.
- hy3 là model suy luận ngầm: chậm (26 s mỗi câu) và tốn nhiều completion token, nên bị loại.
- Từ đây chọn catalog 6 cấu hình trải theo chi phí, lưu ở `configs/real_catalog.json`: `flash_direct`, `pro_direct`, `glm_direct`, `pro_decompose`, `qwen_max_direct`, `kimi_decompose`.

### 2.3 Matrix phản thực tế: 300 câu × 6 cấu hình

Lệnh: `python experiments/run_matrix.py --n 300 --configs catalog --out results/matrix.jsonl`, chạy 16 luồng, mất khoảng 15 phút.

- Chạy **mọi** cấu hình trên **mọi** câu hỏi, tổng cộng 1800 lần gọi.
- Mỗi dòng kết quả ghi: SQL, đúng/sai, token, chi phí, latency.

### 2.4 Đánh giá policy offline

Lệnh: `experiments/evaluate_policies.py`.

**Bằng chứng DB.** Với mỗi candidate, chạy 4 operator, cache trong `cache/evidence/`. Mã nằm ở `plangate/db_evidence.py`:

| operator | làm gì |
|---|---|
| `PARSE_BIND` | chạy `EXPLAIN QUERY PLAN` để bắt lỗi cú pháp, bảng hoặc cột không tồn tại |
| `EXPLAIN` | đặc trưng của query plan: full scan, temp b-tree, join không có điều kiện |
| `VALUE_PROBE` | với mỗi predicate `cột = 'literal'`, kiểm tra `EXISTS`. Nếu literal không có trong DB thì tìm các giá trị gần (LIKE theo cả chuỗi và theo từng từ). |
| `EXEC_BOUNDED` | chạy có timeout, chỉ giữ *hình dạng* kết quả (số dòng, số cột, tỉ lệ NULL) và một fingerprint không phụ thuộc thứ tự dòng |

**Đặc trưng trạng thái** (trong `plangate/state_features.py`), chia đúng theo ablation RQ1:

| mức | gồm những gì |
|---|---|
| P0 | câu hỏi + schema: độ dài, từ khóa, kích thước schema, db one-hot |
| P1 | P0 + đặc trưng của SQL candidate, cộng độ khớp chữ SQL với các candidate trước |
| P2 | P1 + bằng chứng DB, cộng độ khớp **kết quả thực thi** với các candidate trước |

**Các policy được so sánh:** cấu hình cố định, oracle, router một bước (P0), cascade dựa trên luật thực thi, cascade dựa trên độ khớp, và cascade có bộ chấm học được ở P1/P2.

**Kết quả giai đoạn 1** (chi tiết trong `RESULTS.md`):
- Oracle đạt 0.757 với chi phí 3.47, trong khi cấu hình cố định tốt nhất chỉ đạt 0.670 với chi phí 9.74.
- AUROC tăng dần: P0 0.58 → P1 0.64 → P2 0.72.
- Repair dựa trên bằng chứng DB có vẻ hứa hẹn trên các câu bị gắn cờ, nhưng n mới chỉ là 23.

---

## 3. Giai đoạn 2: phân tích theo từng action

**Lý do làm giai đoạn này.** Không nên chạy ngay full dev. Trước hết phải đo **giá trị của từng action riêng lẻ**, và xem đáp án đúng có thực sự cần các action khác nhau ở các trạng thái khác nhau hay không.

### 3.1 Bảng action theo từng câu

Lệnh: `experiments/build_action_table.py`, cho ra `results/action_table.jsonl`.

- **Candidate đầu tiên c0** = `flash_direct`.
- Pool được thu nhỏ còn 3 model (flash, pro, qwen), để kết quả không bị quy cho việc dùng ensemble lớn.
- Các action sau c0:

| action | loại | mô tả |
|---|---|---|
| `VALUE_FIX` | **chỉ dùng DB**, viết mới ở bước này | Nếu một literal không tồn tại trong DB và DB có đúng một giá trị khớp (bỏ qua hoa/thường, hoặc là giá trị gần duy nhất), thì viết lại literal đó mà không cần gọi LLM. Mã: `value_fix()` trong `db_evidence.py`. |
| `REPAIR_DB` | LLM | flash sửa SQL dựa trên bằng chứng DB của c0 |
| `REPAIR_BLIND` | LLM, đối chứng | cùng prompt nhưng phản hồi chung chung, không có thông tin từ DB |
| `RESAMPLE_1`, `RESAMPLE_2` | LLM | flash sinh lại ở temperature 0.8. `sample id` được đưa vào khóa cache để mỗi mẫu là độc lập. |
| `SWITCH_PRO`, `SWITCH_QWEN` | LLM | lấy từ matrix |

- Mỗi SQL sinh ra đều được chấm EX, chạy đủ 4 probe và ghi lại latency của từng probe.
- Tốn khoảng 1.200 lần gọi flash.

### 3.2 Bốn phân tích

Tất cả nằm trong `experiments/analyze_actions.py`.

1. **Oracle action.**
   - Với mỗi câu, tìm action rẻ nhất cho ra đáp án đúng, xét theo thứ tự chi phí trung bình.
   - Tính thêm tỉ lệ cứu được (không loại trừ nhau), chia theo vùng trạng thái: DB đã gắn cờ c0 và chưa gắn cờ.
   - *Gắn cờ* nghĩa là ít nhất một trong các điều kiện sau xảy ra: lỗi parse/bind, join không có điều kiện, literal không tồn tại, lỗi hoặc timeout khi chạy, kết quả rỗng, kết quả toàn NULL.

2. **Experiment A: ΔAUROC theo từng action.**
   - Mục tiêu dự đoán: c0 có đúng không.
   - Baseline là P1. Thêm vào từng nhóm đặc trưng một: từng probe DB riêng, tất cả probe DB, một sample LLM thêm (so chữ SQL hoặc so kết quả thực thi), candidate từ model khác, và "repair không đổi SQL".
   - Mô hình: logistic regression với L2 (C=0.3), lặp 10 lần 5-fold, lấy dự đoán out-of-fold.
   - Khoảng tin cậy: paired bootstrap 2000 lần.
   - Mỗi action còn ghi thêm: tỉ lệ áp dụng được, tỉ lệ gắn cờ, latency median và p95, chi phí.

3. **Experiment C: so action DB với một lần gọi LLM thêm.**
   - Thay c0 bằng output của action, tính ΔP(correct), rồi chia cho số ms hoặc đơn vị chi phí bỏ thêm.
   - Chia theo 3 vùng: bị gắn cờ, không gắn cờ và còn bất định (p < 0.6), không gắn cờ và đã tự tin.

4. **Experiment B: có cần chạy probe này không?**
   - Bộ chấm được train trên **cả 16 tập con probe** (probe nào thiếu thì cờ `has_*` bằng 0), nên quyết định được dựa trên bằng chứng một phần.
   - Một *gate* học được dự đoán "nếu chạy probe này thì quyết định có đổi không". Probe được xét theo thứ tự từ rẻ đến đắt.
   - So 3 policy: không chạy probe nào, chạy tất cả, chạy thích nghi (No / All / Adaptive).
   - Khi phương án dự phòng (fallback) là `REPAIR_DB`, chi phí của các probe mà repair cần cũng được cộng vào cho công bằng.

**Kết quả** (chi tiết trong `ACTION_RESULTS.md`):
- Bằng chứng DB tăng AUROC +0.080 chỉ trong 20 ms. Một sample LLM thêm mà chỉ so chữ SQL thì bằng 0.
- Action chỉ dùng DB (VALUE_FIX) cứu được rất ít câu: khoảng 1%.
- Ở vùng bị gắn cờ, action rẻ cứu được 50% lỗi. Ở vùng không gắn cờ, chỉ cứu được 18%.
- Thay c0 một cách mù quáng thì **làm giảm** accuracy.
- Adaptive probe giảm 81% thời gian DB, nhưng tác động end-to-end không đáng kể vì trên SQLite, DB vốn đã rất rẻ.

**Quyết định rút ra:**
- Chuyển hướng sang *planning lấy verification làm trung tâm*: DB đóng vai cảm biến.
- Mọi action thay thế phải được kiểm chứng trước khi được chấp nhận.

---

## 4. Giai đoạn 3: planner có ngân sách (Experiment D)

Script: `experiments/planner_d.py`.

### 4.1 Mô hình hóa

| thành phần | định nghĩa |
|---|---|
| Trạng thái | tập S các candidate đã có (luôn chứa c0), kèm bằng chứng DB của chúng |
| Action | ACCEPT, VALUE_FIX, RESAMPLE (R1 trước, rồi R2), REPAIR_DB, SWITCH_PRO, SWITCH_QWEN. Tối đa 3 action. |
| Đáp án cuối | candidate được bộ chọn ưa thích nhất trong S: **verified replacement**, không thay thế mù |
| Mô hình học | A(g\|S) = P(candidate g đúng); V(s) = P(lựa chọn hiện tại đúng); Q(s,a) = P(đúng sau khi làm a rồi chọn lại) |
| Quyết định | chọn a\* = argmax Q(s,a) − λ·cost(a); chỉ làm nếu Q − V > λ·cost, ngược lại thì dừng |

**Cách train:**
- Liệt kê mọi tập con S có tối đa 3 action, có kiểm tra ràng buộc thứ tự (R2 chỉ được có khi đã có R1).
- Nhãn Q được tính bằng cách phát lại thật trên cache: thêm a vào S, chọn lại bằng A, rồi xem kết quả đúng hay sai.
- Mô hình: `HistGradientBoostingClassifier` (max_depth 3, 200 vòng).
- **Cross-fit theo câu hỏi** bằng GroupKFold 5 phần.

**Các biến thể dùng làm ablation DBMS:**

| biến thể | được dùng gì |
|---|---|
| `full` | cả 4 probe, cụm theo fingerprint thực thi, VALUE_FIX, REPAIR_DB |
| `exec_only` | chỉ chạy query có giới hạn |
| `nodb` | không truy cập DB; chỉ dựa vào độ khớp chữ SQL; dùng REPAIR_BLIND |

### 4.2 Baseline (chạy trên cùng cache)

- Model cố định.
- Router một bước P0 (logistic theo từng cấu hình, có cross-fit).
- Cascade theo luật DB: gắn cờ thì chuyển sang qwen.
- Cascade theo độ khớp, kiểu MoT.
- Self-consistency @2 và @3, bỏ phiếu bằng kết quả thực thi.
- Ensemble 3–5 candidate bỏ phiếu bằng kết quả thực thi, tương ứng ngân sách kiểu PRISM.
- Confidence cascade (thay thế, hoặc chọn có kiểm chứng).
- Oracle.

**Latency:** ensemble và self-consistency chạy **song song**, latency = max. Planner chạy **tuần tự**, latency = tổng.

### 4.3 Kết quả lần đầu và chẩn đoán

- **Lần đầu:** thắng rõ ở ngân sách thấp (+2.3 điểm), nhưng bị chững ở 0.653.
- **Chẩn đoán** bằng `experiments/selector_diag.py`:
  - Oracle chọn đúng trong pool đạt 0.763, trong khi bỏ phiếu theo thực thi chỉ đạt 0.663.
  - **Nút thắt nằm ở khâu chọn candidate, không phải khâu sinh.**
  - Khi flash và qwen cho kết quả khác nhau: qwen đúng 27 câu, flash đúng 14 câu.

Bước tiếp theo vì vậy là Experiment J, rồi sửa bộ chọn.

---

## 5. Giai đoạn 4: khâu chọn candidate (Experiment J)

**Ý tưởng.** Cho DBMS tính ra *điểm khác nhau* giữa hai cách hiểu câu hỏi, thay vì đưa cho judge hai đoạn SQL. Đó là **differential execution probe**, viết trong `plangate/diff_probe.py`:
- Lấy hình dạng của hai kết quả.
- Đếm và lấy ví dụ của `A EXCEPT B` và `B EXCEPT A`.
- Tất cả đều có timeout và chạy trên kết nối read-only.

**Thiết kế thí nghiệm** (`experiments/judge_exp.py`):
- Lấy mọi cặp (c0, X) có kết quả thực thi khác nhau: 270 cặp. Trong đó 155 cặp là *quyết định được*, tức đúng một bên đúng.
- Judge phải chọn A hoặc B trong 3 điều kiện: chỉ có SQL; SQL + xem trước kết quả; SQL + differential probe.
- **Kiểm soát lệch vị trí:** mỗi cặp được hỏi theo cả hai thứ tự A/B. Accuracy là trung bình của hai thứ tự (`experiments/judge_stats.py`, bootstrap 4000 lần).
- Hai judge: flash (rẻ) và pro.

**Kết quả:**
- Chỉ nhìn SQL, judge đúng 0.43–0.47, **tệ hơn tung đồng xu**.
- Có bằng chứng DB thì tăng 11–16 điểm.
- Với flash, differential probe tốt hơn xem trước kết quả +5.5 điểm, CI [+0.3, +10.6].

---

## 6. Giai đoạn 5: nhánh PostgreSQL

**Mục đích:**
- Kiểm tra tính tổng quát trên engine thứ hai, với dialect khác và một optimizer cost-based thật.
- Kiểm tra giả thuyết systems: **optimizer có định giá được bằng chứng trước khi phải trả tiền cho nó không?**

### 6.1 Dựng server

```
initdb -D data/pg/cluster -U postgres -A trust -E UTF8 --locale=C
pg_ctl -D data/pg/cluster -o "-p 5544 -c listen_addresses=localhost -c shared_buffers=1GB -c work_mem=64MB" start
CREATE ROLE xiaolongli; CREATE DATABASE bird;   -- dump tham chiếu tới owner này
psql -d bird -f minidev/MINIDEV_postgresql/BIRD_dev.sql ; ANALYZE;
```

- Nạp xong trong 34 giây, không lỗi, được 75 bảng (1 GB).
- Đã kiểm tra trước: tên bảng không trùng nhau giữa 11 database, nên dùng chung một schema `public` được.

### 6.2 Backend PostgreSQL (`plangate/pg.py`)

- **Kết nối:** mỗi luồng một kết nối, `default_transaction_read_only = on`, đặt `statement_timeout` cho từng câu.
- **Schema cho prompt:** sinh từ `information_schema`, có quote tên định danh, kèm 3 dòng mẫu mỗi bảng.
- **Chấm EX:** chuẩn hóa số (Decimal và float làm tròn 4 chữ số) trước khi so khớp tập kết quả. Cả 500/500 gold đều chạy được.
- **Bằng chứng:**
  - `EXPLAIN (FORMAT JSON)` cho ước lượng tổng chi phí, số dòng và loại node (seq scan, index scan, nested loop, hash join, sort, aggregate), mà **không chạy query**.
  - Value probe trên PostgreSQL: `::text` kết hợp `ILIKE`.
  - Query có giới hạn, và differential probe.
- Bằng chứng được cache trong `cache/pg_evidence/`.
- Đặc trưng optimizer được thêm vào `db_features`: `ex_log_cost`, `ex_log_rows`, `ex_est_one_row` và số lượng từng loại operator.

### 6.3 Bảng action PostgreSQL

Lệnh: `experiments/pg_build.py`, cho ra `results/pg_action_table.jsonl`.

- 500 câu mini-dev, prompt viết cho dialect PostgreSQL.
- Các action: c0 (flash), RESAMPLE_1, REPAIR_DB, VALUE_FIX, SWITCH_PRO, SWITCH_QWEN.
- Khoảng 2.500 lần gọi, mất khoảng 17 phút.

### 6.4 Phân tích (`experiments/pg_analysis.py`)

- **P1:** lặp lại oracle và Experiment A trên PostgreSQL. Kết quả lặp lại được.
- **P2: optimizer định giá.**
  - Trên 2.446 candidate đã chạy, tính Spearman giữa log ước lượng chi phí và log thời gian chạy thật: **0.962**.
  - AUROC để dự đoán query chạy quá 100 ms là 0.924; quá 500 ms là 0.999.
- **P3: thu thập bằng chứng có tính giá.**
  - Chỉ chạy c0 khi ước lượng chi phí ≤ phân vị p50/p75/p90/p95; nếu không thì chỉ dùng các probe rẻ.
  - So với: luôn chạy, không bao giờ chạy, và chạy ngẫu nhiên 75%/90% (đối chứng).
  - Ở ngưỡng p95: AUROC giữ nguyên (0.759 so với 0.758), thời gian DB giảm 21%, thời gian tối đa giảm 42%.

---

## 7. Giai đoạn 6: hoàn thiện planner và kiểm định

### 7.1 Bộ chọn pairwise

Đây là cách sửa trực tiếp cho F3.
- Gom các candidate thành cụm theo fingerprint thực thi. Biến thể no-DB thì gom theo chữ SQL đã chuẩn hóa.
- Mỗi cụm lấy một đại diện là candidate có A cao nhất.
- Mô hình pairwise P(a ≻ b) dùng: hiệu số đặc trưng giữa hai candidate, nguồn sinh ra từng bên, và điểm A của hai bên. Chỉ train trên các cặp mà đúng một bên đúng, có cross-fit.
- Cụm được chọn là cụm có tổng điểm thắng–thua cao nhất.
- **Kết quả:** tốt hơn bộ chọn pointwise (SQLite 0.667 so với 0.653).

### 7.2 Policy hai pha

Policy một bước cũ luôn ưa các action rẻ, tiêu hết ngân sách 3 action và **không bao giờ chuyển sang qwen**. Policy mới (`run_hybrid`):
- **Pha 1:** làm tối đa 2 action rẻ do DB dẫn đường, chọn theo Q − V − λ·cost.
- **Pha 2:** nếu V(s) < τ thì chuyển sang qwen.
- Đáp án cuối luôn là candidate được chọn có kiểm chứng.
- Quét λ ∈ {0.02, 0.05, 0.1, 0.25} × 15 giá trị τ.

### 7.3 So sánh ở cùng ngân sách và split-half

- **Họ policy PlanGate-full** gồm: greedy, hai pha, và phiên bản có tính đến latency.
- **"Baseline ngoài"** loại trừ các confidence cascade, vì chúng dùng lại bộ chấm và bộ chọn của chính PlanGate.
- Ở mỗi ngân sách (1.25× đến 4× chi phí của flash), lấy điểm tốt nhất của từng họ, rồi chạy paired bootstrap 5000 lần trên chênh lệch accuracy.
- **Split-half:**
  - Hoán vị ngẫu nhiên các câu (seed 1) rồi chia đôi.
  - Chọn điểm vận hành (λ, τ, cùng baseline tốt nhất) trên một nửa, chấm trên nửa còn lại, rồi đổi vai hai nửa.
  - Bootstrap trên chênh lệch đã gộp từ cả hai lượt.
  - Mục đích là chống việc PlanGate được lợi vì có nhiều tham số để chỉnh hơn baseline.
- **Bảng "khớp accuracy":** chi phí và p95 latency mà PlanGate cần để bằng accuracy của từng baseline.

**Kết quả cuối** (chi tiết trong `ICDE_STORY.md`):

| | SQLite | PostgreSQL |
|---|---|---|
| Chênh lệch so với baseline ngoài tốt nhất | +2.3 đến +3.0 điểm, có ý nghĩa ở 1.25–2× | +1.6 đến +3.8 điểm, có ý nghĩa ở 1.5–4× |
| Split-half | +3.3 | tới +4.2 |
| Accuracy cao nhất | | 0.676, cao hơn qwen đứng một mình (0.640) |
| Ablation không DB | luôn kém hơn ở cùng ngân sách | luôn kém hơn ở cùng ngân sách |

---

## 8. Các quy ước chung

| Hạng mục | Quy ước |
|---|---|
| **Chi phí LLM** | `rate(model) × (prompt_tokens + 4 × completion_tokens) / 1000`. `rate` là hệ số giá của MiraiAPI; output được tính đắt gấp 4 lần input. |
| **Chi phí DB** | 1 đơn vị cho mỗi giây DB. Luôn báo cáo tách riêng: ms, p95, max. |
| **Latency** | Đo thời gian thực của từng lần gọi LLM cộng thời gian DB. Ensemble tính song song (max), planner tính tuần tự (tổng). |
| **Accuracy** | BIRD EX: so khớp tập dòng kết quả, có tính gold thay thế cho câu hòa (SQLite). PostgreSQL: so tập dòng sau khi chuẩn hóa số. |
| **Chống rò rỉ** | Mọi mô hình học (router, bộ chấm, bộ chọn, V, Q, gate) đều cross-fit theo câu hỏi. Không câu nào được chấm bởi mô hình đã thấy nó lúc train. |
| **Thống kê** | Paired bootstrap trên câu hỏi. Báo CI 95% và P(Δ > 0). Mọi khác biệt dưới khoảng 2 điểm với n = 300 đều coi là nhiễu. |
| **An toàn** | Kết nối read-only, chỉ cho một câu SELECT/WITH duy nhất, có timeout cho từng câu, giới hạn số dòng lấy về. Generated SQL không bao giờ được phép ghi vào DB. |
| **Tái lập** | Mọi phản hồi LLM được cache theo hash; mọi bằng chứng DB được cache theo (op, db, sql). Seed cố định: lấy mẫu = 0, fold = 0, split = 1. |

---

## 9. Bản đồ file và lệnh chạy lại

### Code

```
plangate/
  providers/mirai.py     client MiraiAPI, bảng RATE
  bird.py                BIRD loader, read-only SQLite, chạy có giới hạn, chấm EX (+ gold thay thế)
  llm_gen.py             prompt GEN/REPAIR, cache LLM, tách SQL
  db_evidence.py         PARSE_BIND / EXPLAIN / VALUE_PROBE / EXEC_BOUNDED, VALUE_FIX, phản hồi cho repair
  evidence_cache.py      cache bằng chứng DB (SQLite)
  diff_probe.py          differential execution probe (SQLite)
  pg.py                  backend PostgreSQL: schema, chấm điểm, bằng chứng + định giá bằng EXPLAIN, diff probe
  state_features.py      đặc trưng P0/P1/P2 (+ đặc trưng optimizer PostgreSQL)
  learned_planner.py     planner value-based đời đầu (giai đoạn 1)
experiments/
  run_matrix.py          matrix phản thực tế (pilot / catalog)
  summarize_matrix.py    tóm tắt matrix
  evaluate_policies.py   giai đoạn 1: E5, RQ1, frontier, bootstrap
  run_repair.py          giai đoạn 1: repair trên các câu bị gắn cờ
  error_analysis.py      phân loại lỗi
  plot_report.py         vẽ frontier giai đoạn 1
  build_action_table.py  bảng action SQLite (giai đoạn 2)
  analyze_actions.py     oracle, Experiment A/B/C
  rescore.py             chấm lại với gold thay thế
  selector_diag.py       chẩn đoán khoảng cách ở khâu chọn
  judge_exp.py           Experiment J
  judge_stats.py         CI cho Experiment J
  planner_d.py           Experiment D: planner, baseline, ablation, split-half (sqlite | pg)
  pg_build.py            bảng action PostgreSQL
  pg_analysis.py         PostgreSQL: Experiment A, định giá bằng optimizer, thu thập bằng chứng có tính giá
configs/real_catalog.json  6 cấu hình sinh SQL
```

### Kết quả

| file | nội dung |
|---|---|
| `RESULTS.md` | giai đoạn 1 |
| `ACTION_RESULTS.md` | giai đoạn 2 |
| `ICDE_STORY.md` | story tổng hợp |
| `matrix.jsonl`, `action_table.jsonl`, `pg_action_table.jsonl` | dữ liệu theo từng câu |
| `judge_*.jsonl` | dữ liệu Experiment J |
| `*.txt` | output thô của từng script |
| `planner_d_report*.json`, `pg_report.json`, `action_report.json`, `offline_report.json` | số liệu dạng máy đọc |
| `frontier.png`, `planner_d_frontier.png`, `planner_d_frontier_pg.png` | hình |

### Chạy lại

Cần có `MIRAI_API_KEY`. Nếu cache còn nguyên thì các bước LLM không tốn thêm tiền.

```
# SQLite
python experiments/run_matrix.py --n 300 --configs catalog --out results/matrix.jsonl
python experiments/rescore.py
python experiments/build_action_table.py results/matrix.jsonl
python experiments/analyze_actions.py
python experiments/judge_exp.py deepseek-v4-flash ; python experiments/judge_exp.py deepseek-v4-pro
python experiments/judge_stats.py
python experiments/planner_d.py sqlite

# PostgreSQL
data/pg/pgsql/bin/pg_ctl -D data/pg/cluster -o "-p 5544" start
python experiments/pg_build.py 500
python experiments/pg_analysis.py
python experiments/planner_d.py pg
```

---

## 10. Các lỗi đã gặp và cách xử lý

| Vấn đề | Nguyên nhân | Cách xử lý |
|---|---|---|
| Key API không có trong tiến trình | Biến được đặt ở cấp User, tiến trình đã chạy từ trước nên không thừa kế | Đọc key từ registry `HKCU\Environment` |
| Pilot bị treo ở một lần gọi | Lần gọi bị treo, cộng với `ex.map` trả kết quả theo thứ tự nên chặn cả hàng | Giảm timeout và retry; chuyển sang `as_completed` |
| `PermissionError` khi ghi cache | Hai luồng cùng `os.replace` vào một khóa (SQL giống hệt nhau từ hai cấu hình) | Thêm lock riêng cho từng khóa |
| Planner value-based đời đầu không sinh thêm candidate | Quy tắc `Q − p_best` không bao giờ dương, vì các model có accuracy biên tương tự nhau | Dùng "giá trị cứu" `(1 − p_best)·Q`, rồi sau đó thay bằng planner Q/V có cross-fit |
| Planner bị chững ở 0.653 | Khâu chọn yếu, và policy một bước không bao giờ chuyển sang model mạnh | Thêm bộ chọn pairwise và policy hai pha |
| Thống kê "diff applicable" ra 0% | Trùng khóa: mode tên `"diff"` ghi đè lên payload diff | Đổi tên thành `diff_payload`. Accuracy của judge vẫn hợp lệ vì prompt được dựng trước khi bị ghi đè. |
| `minidev.zip` bị hỏng | Giải nén khi file còn đang tải dở | Tải tiếp bằng `curl -C -` |
| Không có PostgreSQL hoặc Docker | Máy không cài sẵn; `pgserver` không có wheel cho Python 3.13 | Dùng bản binary zip PostgreSQL 16 của EDB, chạy `initdb` cục bộ |
| Exit code 255 giả | Pipe của PowerShell (`Select-Object -First`) đóng sớm | Ghi output ra file trước rồi mới đọc |
| Mức tăng có thể do chỉnh tham số trên chính tập test | PlanGate có nhiều tham số để chỉnh hơn baseline | Thêm kiểm định split-half |
