# Experiment F: planner tự cải thiện theo luồng câu hỏi (prequential)

Script: `experiments/prequential.py [sqlite|pg] [n_seeds]`.
Output: `results/prequential[_pg].{txt,json,png}`.

## Thiết lập

- Câu hỏi đến theo thứ tự ngẫu nhiên, 5 seed. Planner được huấn luyện lại sau mỗi khối 30 câu và **chỉ học từ các câu đã qua**: không nhìn trước, không cross-fit trên tập đánh giá.
- Toàn bộ được phát lại từ action table đã cache, không có lời gọi LLM mới.
- Điểm vận hành cố định: λ = 0.05, τ = 0.5 (two-phase, chọn lựa có kiểm chứng).
- **Khởi động lạnh:** khi chưa đủ dữ liệu (dưới 8 câu có log), hệ thống dùng rule cascade: nếu có DB trigger thì chuyển sang qwen.
- Các chế độ phản hồi:

| Chế độ | Được học từ | Nhãn |
|---|---|---|
| `static` | không học, rule cascade suốt luồng | – |
| `gold-full` | mọi câu đã qua, mọi hành động (phản thực tế) | gold. Cận trên lý tưởng |
| `logs, gold` | log của chính hệ thống: chỉ các ứng viên đã thực sự chạy, cộng 10% thăm dò thêm 1 hành động ngẫu nhiên (có tính phí) | gold trên các ứng viên đã chạy, tức phản hồi đúng/sai đến trễ |
| `logs, self` | như trên | tự sinh: đúng khi kết quả trùng cụm kết quả đa số |
| `boot30+logs, *` | 30 câu đầu chạy tất cả hành động (có tính phí), sau đó học từ log | gold / tự sinh |
| `explore+self` | 10% câu chạy tất cả hành động (có tính phí) | tự sinh |

## Kết quả (trung bình 5 seed)

Bảng dưới là 1/3 cuối luồng, tức sau khi đã học. `static` là mốc so sánh.

| Chế độ | SQLite EX | cost | PostgreSQL EX | cost |
|---|---|---|---|---|
| static | 0.660 | 3.92 | 0.623 | 3.55 |
| **gold-full** | **0.690** | 5.54 | **0.652** | 5.87 |
| logs, gold | 0.664 | 7.32 | 0.647 | 7.78 |
| boot30+logs, gold | 0.652 | 6.98 | 0.646 | 7.63 |
| logs, self | 0.654 | 3.53 | 0.612 | 3.58 |
| boot30+logs, self | 0.650 | 3.75 | 0.618 | 3.79 |
| explore+self ε=0.1 | 0.664 | 6.64 | 0.619 | 5.21 |
| *offline cross-fit (Exp D, toàn bộ)* | *0.670* | *5.85* | *0.660* | *6.57* |

- **Có phản hồi đầy đủ thì vòng học hoạt động.**
  - `gold-full` cao hơn `static` +3.0 điểm (SQLite) và +2.9 điểm (PostgreSQL) ở 1/3 cuối luồng.
  - Mức gain tích lũy tăng dần theo số câu đã phục vụ, và cost mỗi câu giảm dần (SQLite: 6.6 xuống 5.2).
  - Ở 1/3 cuối, kết quả bằng hoặc cao hơn planner cross-fit offline. Vậy đánh giá prequential có thể thay cho cross-fit.
- **Học từ log của chính mình chỉ đạt một phần mức tăng đó.**
  - `logs, gold`: +0.4 điểm (SQLite) và +2.4 điểm (PostgreSQL), nhưng tốn khoảng gấp đôi cost.
  - Nguyên nhân: log chỉ chứa những hành động mà policy đã chọn (thiên lệch off-policy). Planner không thấy được hành động rẻ mà nó chưa từng thử, nên có xu hướng leo thang lên model mạnh.
- **Nhãn tự sinh không giúp gì, thậm chí làm kém đi.**
  - `logs, self` đạt −0.6 điểm (SQLite) và −1.1 điểm (PostgreSQL) so với `static`.
  - Nhãn đồng thuận chỉ khớp gold khoảng 72%. Độ chính xác của nhãn "đúng" là 0.70–0.72, gần tỉ lệ nền 0.61–0.67.
  - Khi các model cùng sai một kiểu, đồng thuận vẫn chọn đáp án sai.
  - Ngoài ra, các câu ACCEPT chỉ có một ứng viên nên không tạo ra nhãn. Log vì vậy chỉ gồm những câu đã leo thang.
- **L1 (kho evidence dùng lại giữa các câu):**
  - Truy vấn VALUE_PROBE trùng với câu trước khá thường xuyên: 30–66% số probe đã từng gặp.
  - Khóa SQL nguyên văn hiếm khi trùng: 1–20%.
  - Thời gian DB tiết kiệm được là 3–21%. Vì chi phí DB tính bằng mili giây nên gần như không ảnh hưởng tới cost; lợi ích nằm ở độ trễ và tải DB.

Lưu ý: 1/3 cuối luồng chỉ gồm khoảng 100 câu (SQLite) và 167 câu (PostgreSQL). 5 seed là 5 thứ tự khác nhau của **cùng** một tập câu hỏi, nên chênh lệch dưới khoảng 2 điểm chưa thể coi là có ý nghĩa thống kê.

## Experiment F2: học được từ log của chính hệ thống

Script: `experiments/prequential2.py [sqlite|pg] [n_seeds]`.
Output: `results/prequential2[_pg].{txt,json,png}`.

Ba thay đổi so với F:
1. **Advantage model.** Ở F, Q(s,a) và V(s) được học trên hai tập câu khác nhau: V trên mọi câu, Q chỉ trên những câu từng chạy a. Vì vậy Q − V lệch dương và planner làm thừa hành động. ADV(s,a) hồi quy trực tiếp mức lợi Δ = đúng(select(S+a)) − đúng(select(S)) trên cùng một câu hỏi. Planner chọn hành động rẻ khi ADV − λ·C > 0.
2. **Nhãn tự sinh v2.** Quy tắc gán nhãn:
   - **Sai** nếu DB xác nhận lỗi: lỗi bind, lỗi thực thi, kết quả rỗng hoặc toàn NULL, hoặc literal không tồn tại. Các nhãn này đúng 77–100% theo từng loại.
   - **Đúng** nếu ứng viên sạch về DB và có ít nhất 2 họ model cho cùng kết quả.
   - Còn lại thì bỏ, không gán nhãn.
3. **Audit chủ động.** Mỗi khối có một tỉ lệ β câu nhận phản hồi đúng/sai. Câu được chọn audit là câu có V(s) cuối gần 0.5 nhất (uncertainty sampling), so với cách chọn ngẫu nhiên.

Kết quả ở 1/3 cuối luồng, trung bình 5 thứ tự. Cột cuối là chênh lệch EX so với `static`, bootstrap ghép cặp theo câu hỏi, khoảng tin cậy 95%.

| Chế độ | nhãn/câu | SQLite EX / cost | vs static | PostgreSQL EX / cost | vs static |
|---|---|---|---|---|---|
| static | 0 | 0.660 / 3.92 | – | 0.623 / 3.55 | – |
| gold-full (Q−V) | 1.00 | 0.690 / 5.54 | +3.0 [+1.4, +4.6] | 0.652 / 5.87 | +2.9 [+1.3, +4.5] |
| gold-full (ADV) | 1.00 | 0.688 / 5.59 | +2.8 [+1.2, +4.4] | 0.657 / 5.74 | +3.4 [+1.8, +4.9] |
| logs, gold 100% (Q−V) | 1.00 | 0.666 / 7.41 | +0.6 [−1.2, +2.6] | 0.655 / 7.61 | +3.3 [+1.3, +5.3] |
| logs, gold 100% (ADV) | 1.00 | 0.666 / 7.04 | +0.6 [−1.4, +2.8] | 0.640 / 7.14 | +1.7 [−0.2, +3.6] |
| logs, self-v2 (ADV) | **0** | 0.668 / 4.85 | **+0.8 [+0.2, +1.6]** | 0.624 / 4.21 | +0.1 [−0.2, +0.6] |
| + audit 10%, uncertainty | 0.10 | 0.660 / 5.39 | +0.0 [−1.2, +1.2] | 0.639 / 5.31 | +1.6 [+0.4, +2.9] |
| **+ audit 25%, uncertainty** | 0.27 | **0.674 / 6.47** | **+1.4 [+0.2, +2.8]** | **0.649 / 5.93** | **+2.7 [+1.1, +4.2]** |
| + audit 25%, ngẫu nhiên | 0.27 | 0.668 / 6.44 | +0.8 [−0.4, +2.2] | 0.627 / 6.30 | +0.4 [−1.2, +1.9] |

- **Nhãn tự sinh v2 không còn gây hại.**
  - Ở F, nhãn tự sinh từ đồng thuận cho −0.6 và −1.1 điểm so với static. Với v2 cộng ADV, kết quả là +0.8 (có ý nghĩa thống kê) và +0.1.
  - Không cần bất kỳ nhãn nào từ bên ngoài. Cost tăng ít (4.2–4.9) và ACCEPT ở mức 0.86–0.89.
- **Audit chủ động trên 25% số câu đạt gần mức của phản hồi đầy đủ.**
  - PostgreSQL: +2.7, so với +3.4 của gold-full. SQLite: +1.4, so với +2.8.
  - Đường gain tích lũy vẫn còn đi lên ở cuối luồng (`prequential2_pg.png`).
- **Chọn đúng câu để audit quan trọng hơn số lượng audit.** Cùng ngân sách 25%, chọn theo độ bất định cho +1.4 / +2.7, còn chọn ngẫu nhiên chỉ được +0.8 / +0.4.
- **ADV giảm việc làm thừa.**
  - Khi học từ log với gold 100%, ACCEPT tăng từ 0.49 lên 0.66 (SQLite) và từ 0.44 lên 0.59 (PostgreSQL), cost giảm khoảng 0.4–0.5.
  - Tuy nhiên accuracy không tăng theo, và trên PostgreSQL còn giảm 1.6 điểm. Vậy ADV chưa phải cải tiến rõ ràng về độ chính xác.
- **Học từ log vẫn tốn cost hơn gold-full** (khoảng 6–7 so với 5.5–5.9). Thiên lệch off-policy mới chỉ được giảm một phần.
- **Lưu ý:** audit 10% trên SQLite cho +0.0, và các hiệu ứng dưới 1 điểm còn dao động theo thứ tự câu hỏi. Cần một luồng dài hơn (1534 câu dev) để khẳng định chắc chắn.

## Kết luận cho story

1. Caching tín hiệu để planner tự học là khả thi. **Nút thắt là nguồn nhãn, không phải bộ nhớ cache.**
2. Đóng góp có thể trình bày (C4): **vòng phản hồi evidence**, theo mô hình LEO/Bao. Có ba thành phần:
   - Nhãn âm do DB xác nhận.
   - Nhãn dương do nhiều họ model đồng thuận.
   - Audit chủ động trên khoảng 1/4 số câu.
   Kết quả đạt 50–80% mức tăng của phản hồi đầy đủ (F2).
3. Hướng tiếp theo:
   1. Luồng dài hơn: toàn bộ 1534 câu dev, cần gọi API cho các câu chưa có trong cache.
   2. Hiệu chỉnh off-policy (inverse propensity) để giảm cost của chế độ học từ log.
   3. Audit bằng model mạnh nhất (judge) thay cho phản hồi đúng/sai. Đo xem giảm bao nhiêu so với audit bằng gold.
   4. L3: dùng lại đáp án đã kiểm chứng làm few-shot. Bước này cần gọi API.
