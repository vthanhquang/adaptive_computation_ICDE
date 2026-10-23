# So sánh với phương pháp đã công bố

**Phạm vi:** AutoMix, CHASE-SQL selection, EllieSQL và SQLens, được cài đặt lại trong cùng setting của PlanGate.

**Script:**
- `experiments/paper_baselines.py`: chạy hai phương pháp.
- `experiments/planner_d.py`: so sánh ở cùng ngân sách và kiểm định split-half.

Output thô nằm trong `results/paper_baselines*.txt` và `results/planner_d*.txt`.

## Phương pháp được cài đặt lại

### AutoMix (Aggarwal et al., NeurIPS 2024), biến thể meta-verifier dạng ngưỡng

- Model nhỏ (flash) sinh SQL.
- Chính model đó tự kiểm chứng câu trả lời 3 lần, lấy mẫu ở T=0.7 và **không thực thi** câu SQL.
- Confidence là tỉ lệ số lần trả lời "Yes".
- Nếu confidence < θ thì chuyển sang model lớn (qwen); câu trả lời của qwen thay thế câu trả lời của flash.
- **Khác bản gốc:** prompt kiểm chứng là zero-shot (bài gốc dùng few-shot), và không dùng meta-verifier POMDP.

### CHASE-SQL selection (Pourreza et al., ICLR 2025)

- Nhiều generator sinh candidate; các candidate được gom cụm theo kết quả thực thi.
- Một selection agent so sánh từng cặp đại diện cụm, dựa trên câu hỏi, schema và kết quả thực thi của cả hai bên.
- Candidate thắng nhiều cặp nhất được chọn.
- **Khác bản gốc:**
  - Generator là pool hiện có (flash, pro, qwen, và thêm resample của flash) thay vì ba generator CoT của CHASE.
  - Selector là deepseek-v4-pro dùng nguyên bản; CHASE fine-tune selector.

Mọi phương pháp được chạy trên cùng cache candidate, cùng cách tính chi phí (token có trọng số + thời gian DB) và cùng cách tính latency. Bước kiểm chứng của AutoMix và các phép so sánh cặp của CHASE được tính là chạy song song.

## Kết quả

| Phương pháp | SQLite EX | SQLite chi phí | SQLite p95 (s) | PostgreSQL EX | PostgreSQL chi phí | PostgreSQL p95 (s) |
|---|---|---|---|---|---|---|
| AutoMix θ=0.3 | 0.647 | 12.91 | 48.1 | 0.602 | 12.72 | 42.9 |
| AutoMix θ=0.6 | 0.650 | 13.76 | 56.2 | 0.614 | 13.68 | 49.6 |
| AutoMix θ=0.9 | 0.667 | 15.11 | 62.6 | 0.632 | 14.99 | 54.1 |
| CHASE-SQL (flash, pro, qwen) | 0.637 | 21.42 | 44.5 | 0.620 | 21.46 | 43.3 |
| CHASE-SQL (flash×2, pro, qwen) | 0.637 | 25.57 | 54.0 | 0.624 | 26.02 | 46.5 |
| **PlanGate (điểm tốt nhất)** | **0.680** | 9.52 | 45.3 | **0.676** | 11.09 | 32.3 |

Chi phí và p95 latency mà PlanGate cần để đạt cùng accuracy:

| Phương pháp | SQLite chi phí | SQLite p95 | PostgreSQL chi phí | PostgreSQL p95 |
|---|---|---|---|---|
| AutoMix θ=0.3 / 0.6 / 0.9 | −74% / −76% / −75% | −32% / −42% / −46% | −77% / −75% / −69% | −71% / −72% / −69% |
| CHASE-SQL (3 / 4 generator) | −85% / −88% | −27% / −40% | −83% / −85% | −68% / −71% |

## Nhận xét

- **Kiểm chứng của AutoMix kém hơn bằng chứng DB.**
  - Tự kiểm chứng không thực thi chỉ đạt AUROC 0.624 (SQLite) và 0.651 (PostgreSQL) khi dự đoán candidate đầu tiên đúng hay sai.
  - Bộ chấm dùng bằng chứng DB đạt 0.715.
  - Ba lần kiểm chứng, mỗi lần chứa toàn bộ schema, tốn gần bằng ba lần sinh SQL. Chi phí của AutoMix vì vậy cao gấp khoảng 4–5 lần flash, trong khi accuracy không vượt qwen.
- **CHASE-SQL selection chưa vượt được bỏ phiếu theo kết quả thực thi.**
  - Với selector không fine-tune, accuracy đạt 0.637 (SQLite) và 0.620–0.624 (PostgreSQL), thấp hơn ensemble bỏ phiếu theo thực thi (0.657 / 0.630).
  - Điều này khớp với Experiment J: judge LLM yếu nhất ở các cặp flash–qwen.
  - Kết quả này nhiều khả năng đánh giá thấp CHASE-SQL bản gốc, vốn có selector fine-tune và generator mạnh hơn.
- **Ở mức chi phí của hai phương pháp này (13–26 đơn vị), PlanGate cao hơn từ 1.3 đến 5.6 điểm.**
  - PlanGate dùng chưa tới một nửa chi phí đó: điểm tốt nhất ở 9.5 (SQLite) và 11.1 (PostgreSQL).
  - Trong phép so sánh ở cùng ngân sách 1.25–4×, cả hai phương pháp đều nằm ngoài vùng ngân sách, nên baseline mạnh nhất vẫn là các cascade đơn giản. Bảng bootstrap trong `planner_d*.txt` không đổi so với trước.

## EllieSQL (Zhu et al., 2025, arXiv 2503.22402)

Cài lại sát code công bố (github.com/HKUSTDial/EllieSQL), trong `plangate/elliesql.py` và `experiments/run_elliesql.py`.

- **Prompt:** giữ nguyên văn cho schema linking, G_B (sinh trực tiếp), G_M (divide-and-conquer + refiner dựa trên thực thi) và G_A (5 ví dụ tổng hợp online + divide-and-conquer + refiner).
- **Schema:** schema "enriched" dựng từ `database_description/*.csv` của BIRD, kèm khóa chính và khóa ngoại.
- **Router:** dùng **checkpoint gốc** Qwen2.5-0.5B DPO (`derrickzhu/EllieSQL_Router_Checkpoints/qwen_dpo`), input theo đúng template `create_classifier_prompt`, chạy trên CPU (khoảng 1.3 s mỗi câu).
- **Khác biệt duy nhất:** backbone là deepseek-v4-flash thay cho gpt-4o-mini.
- **Phạm vi:** chỉ chạy nhánh SQLite (300 câu), vì router được train trên BIRD SQLite và các prompt viết riêng cho SQLite.

**Kiểm tra độ trung thực:** router phân bố 53% G_B, 11% G_M, 36% G_A, khớp với tỉ lệ nhãn lúc train trong paper (53% / 9% / 38%). Xu hướng cũng giống paper: router rẻ hơn G_A 24% và mất 0.7 điểm.

| Policy | EX | Chi phí | Token | p95 (s) | Số lần gọi LLM |
|---|---|---|---|---|---|
| EllieSQL G_B (luôn dùng) | 0.593 | 5.68 | 10532 | 38.0 | 2.00 |
| EllieSQL G_M (luôn dùng) | 0.640 | 6.83 | 11722 | 44.6 | 2.01 |
| EllieSQL G_A (luôn dùng) | 0.640 | 9.52 | 15801 | 65.3 | 3.02 |
| EllieSQL + router DPO | 0.633 | 7.21 | 12610 | 56.8 | 2.37 |
| EllieSQL + router oracle | 0.700 | 5.91 | 10820 | 41.8 | 2.04 |
| **PlanGate** | **0.670** | **4.06** | — | 34.1 | 1.19 |

**So sánh trực tiếp** (điểm tốt nhất của PlanGate với chi phí ≤ chi phí của phương pháp; paired bootstrap):

| Phương pháp | PlanGate − phương pháp [CI 95%] | Chi phí PlanGate cần để bằng accuracy | p95 latency PlanGate |
|---|---|---|---|
| EllieSQL + router DPO | +3.7 [−0.7, +8.0] | −57% | −43% |
| EllieSQL G_A | +3.3 [−0.7, +7.7] | −67% | −50% |
| EllieSQL G_B | +7.7 [+3.7, +12.0] | −45% | −14% |
| AutoMix θ=0.3 | +3.3 [+0.3, +6.7] | −74% | −32% |
| CHASE-SQL | +4.3 [+0.7, +8.0] | −85% | −27% |

**Chẩn đoán router** (`results/elliesql_router_diag.txt`):
- Router dự đoán "pipeline cơ bản sẽ sai" với AUROC 0.613, tương đương router P0 (0.58) và thấp hơn bằng chứng DB sau khi sinh SQL (0.715).
- Router chỉ khớp nhãn oracle ở 56% số câu giải được.
- Router oracle đạt 0.700 ở chi phí 5.91. Như vậy dư địa của EllieSQL là có thật, nhưng quyết định trước khi sinh SQL không khai thác được.

**Nhận xét:**
- PlanGate đạt accuracy cao hơn EllieSQL ở chi phí thấp hơn khoảng 44%, và p95 latency thấp hơn 43%.
- Khác biệt accuracy so với router DPO chưa có ý nghĩa thống kê ở n = 300; khác biệt về chi phí thì rõ.
- Một phần chi phí của EllieSQL đến từ bước schema linking: G_B gồm 2 lần gọi LLM, trong khi PlanGate dùng thẳng schema kèm dòng mẫu.

## SQLens (Gong et al., NeurIPS 2025, arXiv 2506.04494)

Cài lại theo mô tả trong paper (`plangate/sqlens.py`, `experiments/run_sqlens.py`). Code gốc chỉ có trên anonymous.4open.science nên không được dùng.

- **Phát hiện lỗi:**
  - 9 tín hiệu từ DB theo định nghĩa ở phụ lục B: join tree thừa bảng (so với Steiner tree trên đồ thị khóa ngoại), join predicate sai khóa, predicate rỗng, kết quả bất thường, subquery vô hướng trả nhiều dòng, GROUP BY không có aggregate, quá 3 subquery, giá trị mơ hồ, bảng tương tự.
  - 5 tín hiệu từ LLM.
  - Gộp tín hiệu bằng một label model sinh, fit không cần nhãn trên chính tập dev (weak supervision), rồi train một classifier trên nhãn mềm của label model.
- **Sửa lỗi** (Algorithm 1): Error Selector, rồi Error Fixer có kiểm tra cú pháp, tối đa 2 vòng, sau đó SQL Auditor chọn giữa SQL gốc và SQL đã sửa.
- **SQL gốc:** candidate flash (c0), giống PlanGate.
- **LLM:** chạy hai biến thể là flash và pro; bản gốc dùng Claude 3.5 Sonnet.
- **Khác bản gốc:**
  - 5 tín hiệu LLM được gộp vào một lần gọi, rẻ hơn bản gốc nên có lợi cho baseline.
  - Value ambiguity chỉ kiểm tra khớp chính xác trên các bảng trong truy vấn và các bảng kề theo khóa ngoại, không có bước LLM đánh giá cột nào "gần nghĩa hơn".
  - Chỉ chạy nhánh SQLite.

| | SQLens (flash) | SQLens (pro) | PlanGate |
|---|---|---|---|
| AUROC phát hiện lỗi ở c0 | 0.666 | 0.646 | ≈0.72 (bộ chấm dùng bằng chứng DB) |
| Tỉ lệ bị gắn cờ / precision / recall | 9.0% / 0.667 / 0.162 | 7.3% / 0.727 / 0.144 | — |
| EX cuối | 0.643 | 0.633 | 0.670–0.680 |
| Chi phí | 9.43 | 13.22 | 4.06–9.52 |
| p95 latency (s) | 72.8 | 59.8 | 34.1–45.3 |
| Thời gian tín hiệu DB, median / p95 | 33 ms / 8.6 s | | |

**So sánh trực tiếp** (điểm tốt nhất của PlanGate với chi phí ≤ chi phí của SQLens):

| Biến thể | PlanGate − SQLens [CI 95%] | Chi phí PlanGate cần để bằng accuracy | p95 latency PlanGate |
|---|---|---|---|
| SQLens (flash) | +3.0 [+0.0, +6.0] | −66% | −55% |
| SQLens (pro) | +4.7 [+1.7, +7.7] | −76% | −46% |

**Nhận xét:**
- SQLens và PlanGate đều dùng tín hiệu DB, nhưng theo hai cách khác nhau.
  - SQLens chạy **toàn bộ** bộ tín hiệu cố định trên mọi truy vấn. Riêng các tín hiệu quét dữ liệu (value ambiguity) khiến p95 thời gian DB lên tới 8.6 s.
  - PlanGate chọn bằng chứng theo trạng thái, định giá trước bằng optimizer, và dùng bằng chứng thực thi để chọn giữa các candidate.
- Từng tín hiệu riêng lẻ của SQLens chỉ có AUROC 0.50–0.60. Gộp lại đạt 0.65–0.67, vẫn thấp hơn bằng chứng dựa trên thực thi và độ khớp kết quả của PlanGate.
- Bộ phát hiện chỉ gắn cờ 7–9% số câu, với recall khoảng 15%, nên phần sửa lỗi chỉ thêm khoảng 1 điểm so với flash gốc. Paper gốc báo mức tăng lớn hơn trên các hệ thống nền yếu hơn (DIN-SQL, MAC-SQL), nên kết quả này phụ thuộc vào chất lượng SQL gốc.

## DIN-SQL và MAC-SQL: đã cài đặt, chưa chạy xong

Code nằm ở `plangate/classic_pipelines.py` và `experiments/run_classic.py`. Prompt được đọc nguyên văn từ file gốc trong `third_party/`.

- **DIN-SQL** (bản BIRD):
  1. Schema linking.
  2. Phân loại câu hỏi thành EASY / NON-NESTED / NESTED.
  3. Sinh SQL theo prompt của từng lớp.
  4. Self-correction.
- **MAC-SQL:** Selector cắt schema (khi DB lớn), Decomposer tách sub-question, Refiner sửa dựa trên kết quả thực thi; tổng cộng tối đa 3 lượt agent.

**Trạng thái:** quota API MiraiAPI đã hết giữa chừng (lỗi 403 `token quota is not enough`). Chỉ 29/300 câu DIN-SQL và 20/300 câu MAC-SQL chạy xong. Kết quả sơ bộ trên các tập nhỏ này:

| | Số câu | EX | Flash c0 trên cùng các câu | Chi phí trung bình |
|---|---|---|---|---|
| DIN-SQL | 29 | 0.483 | 0.483 | 22.4 |
| MAC-SQL | 20 | 0.600 | 0.500 | 7.0 |

Hai tập này quá nhỏ để kết luận nên chưa được đưa vào `paper_baselines.json`. Các lời gọi đã thành công nằm trong cache. Sau khi nạp thêm quota, chạy lại `python experiments/run_classic.py` để hoàn tất, rồi chạy `python experiments/planner_d.py sqlite` để có so sánh ở cùng ngân sách.

## Hạn chế của phép so sánh

- Đây là bản cài đặt lại trong cùng pool model, không phải code gốc. Prompt kiểm chứng của AutoMix là zero-shot, và selector của CHASE-SQL không được fine-tune.
- Chưa có khoảng tin cậy riêng cho phép so sánh trực tiếp với hai phương pháp này, vì kết quả theo từng câu của PlanGate chưa được lưu tại đúng các mức chi phí đó.
- EllieSQL mới chạy trên SQLite. Bản PostgreSQL cần train lại router hoặc chuyển prompt sang dialect PostgreSQL.
- SQLens cũng mới chạy trên SQLite.
- Cần chạy thêm cascade kiểu LOTUS có bảo đảm accuracy; PRISM nếu có code công khai.
