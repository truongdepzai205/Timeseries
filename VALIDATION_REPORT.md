# BÁO CÁO KIỂM TRA KỸ THUẬT

Ngày kiểm tra: 18/06/2026.

## Phạm vi đã kiểm tra

### 1. Kiểm tra cú pháp

- Toàn bộ file Python đã qua `python -m compileall`.
- Hai notebook đã được đọc bằng `nbformat`.
- Tất cả code cell trong notebook đã qua `ast.parse`.

### 2. Smoke test mô hình

`python scripts/smoke_test.py` đã kiểm tra thành công:

- augmentation weak/strong cho MFCC;
- encoder nhận `[N, 13, 96]`;
- encoder trả đặc trưng `[N, 128, 14]`;
- Temporal Contrasting hai chiều;
- NT-Xent contextual loss;
- supervised contrastive loss;
- loss hữu hạn;
- lan truyền ngược của nhánh TS-TCC;
- lan truyền ngược của nhánh CA-TCC/SupCon;
- dataloader đọc đúng payload `.pt`.

### 3. Kiểm tra end-to-end TS-TCC và CA-TCC

Pipeline thu nhỏ bằng dữ liệu tensor giả đã chạy qua:

```text
self_supervised
→ ft_1p (TS-TCC)
→ ft_1p trung gian cho CA-TCC
→ gen_pseudo_labels_1p
→ SupCon_pseudo_1p
→ train_linear_SupCon_pseudo_1p
→ aggregate_results.py
```

Các checkpoint và metric được truyền đúng giữa các giai đoạn. Pseudo-label file giữ nhãn thật ở các `source_indices` đã biết và gắn cờ `is_human_label`.

### 4. Kiểm tra các ablation

Các đường chạy sau đã hoàn thành một epoch với dữ liệu giả:

- `supervised`;
- `ts_pl`;
- `ts_supcon`;
- `ca_scratch`.

Kết hợp với kiểm tra end-to-end ở trên, toàn bộ sáu phương pháp trong ma trận đã đi qua các code path tương ứng.

### 5. Kiểm tra điều phối thí nghiệm

Dry-run đã sinh đúng lệnh cho:

```text
6 phương pháp × 3 tỷ lệ nhãn = 18 cấu hình
```

Script hỗ trợ:

- checkpoint dùng chung cho TS-TCC self-supervised theo seed;
- pseudo-label file riêng theo method, tỷ lệ nhãn và seed;
- tiếp tục lần chạy bị gián đoạn;
- `--force` để chạy lại;
- `orchestration_log.csv` để lưu trạng thái và thời gian;
- đường dẫn data/log tùy chỉnh;
- giới hạn số luồng CPU.

### 6. Kiểm tra tổng hợp kết quả

`aggregate_results.py` đã được chạy trên kết quả giả và tạo thành công:

- `all_runs.csv`;
- `summary_by_method_and_ratio.csv`;
- `macro_f1_pivot.csv`;
- `macro_f1_comparison.png`;
- bảng thời gian từng pipeline khi có orchestration log.

## Những gì chưa được thực hiện trong môi trường đóng gói

- Chưa huấn luyện đủ 40 epoch trên toàn bộ 8.800 mẫu thật.
- Chưa chạy đầy đủ 18 cấu hình × 3 seed trên GPU.
- Vì vậy gói dự án không chứa kết quả accuracy/F1 thực nghiệm thật.

Đây là chủ ý để tránh cung cấp số liệu không được kiểm chứng. Người dùng cần chạy notebook tạo dữ liệu thật và chạy ma trận thí nghiệm trên Colab/GPU.

## Kết luận kỹ thuật

Code đã vượt qua kiểm tra cú pháp, shape, forward, backward, dataloader, pseudo-label, checkpoint flow, resume flow, các ablation và tổng hợp metric bằng dữ liệu kiểm thử. Việc còn lại là huấn luyện thực nghiệm trên dữ liệu thật.
