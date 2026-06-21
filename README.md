# CA-TCC cho Spoken Arabic Digits

Dự án thực nghiệm này được xây dựng dựa trên kiến trúc, loss và quy trình huấn luyện trong repo công khai **CA-TCC** của Eldele và cộng sự. Phần mở rộng tập trung vào một miền chuỗi thời gian mới: **nhận dạng chữ số tiếng Ả Rập từ chuỗi MFCC âm thanh**.

## Mục tiêu

- Tách riêng notebook EDA và tiền xử lý dữ liệu.
- Tạo đầu vào `.pt` đúng định dạng `samples`/`labels` của CA-TCC.
- Chạy đồng thời các tỷ lệ nhãn **1%, 5%, 10%**.
- So sánh trực tiếp **TS-TCC** với **CA-TCC**.
- Bổ sung các ablation để phân tích riêng đóng góp của pseudo-label và SupCon.
- Lưu checkpoint, metric, confusion matrix, lịch sử huấn luyện, chất lượng pseudo-label và thời gian từng giai đoạn.

## Cấu trúc chính

```text
CA-TCC-SpokenArabicDigits-Complete/
├── notebooks/
│   ├── 01_EDA_Preprocessing_SpokenArabicDigits.ipynb
│   └── 02_Experiment_Result_Analysis.ipynb
├── data/SpokenArabicDigits/
├── config_files/SpokenArabicDigits_Configs.py
├── dataloader/
├── models/
├── trainer/
├── experiments/
│   ├── run_experiments.py
│   ├── aggregate_results.py
│   ├── experiment_matrix.csv
│   └── experiment_matrix.json
├── scripts/
│   ├── smoke_test.py
│   └── run_full_matrix.sh
├── main.py
├── ca_tcc_pipeline.sh
├── RUN_GUIDE_VI.md
└── VALIDATION_REPORT.md
```

## Dữ liệu mới

Spoken Arabic Digit là dữ liệu time series đa biến. Mỗi mẫu là một chuỗi có 13 hệ số MFCC theo thời gian và thuộc một trong 10 lớp chữ số từ 0 đến 9.

Notebook xử lý:

```text
Chuỗi MFCC có độ dài thay đổi
→ EDA và kiểm tra chất lượng
→ chia train/validation/test theo người nói
→ chuẩn hóa chỉ từ train
→ nội suy về 96 frame
→ chuyển thành [N, 13, 96]
→ tạo train.pt, val.pt, test.pt
→ tạo train_1perc.pt, train_5perc.pt, train_10perc.pt
```

Dữ liệu không được đóng gói sẵn. Notebook tự tải từ UCI khi chạy.

## Các phương pháp được hỗ trợ

| Method | TS-TCC pretrain | Pseudo-label | SupCon | Vai trò |
|---|---:|---:|---:|---|
| `supervised` | Không | Không | Không | Baseline ít nhãn |
| `ts_tcc` | Có | Không | Không | TS-TCC riêng |
| `ts_pl` | Có | Có | Không | Ablation pseudo-label |
| `ts_supcon` | Có | Không | Có, nhãn thật | Ablation SupCon |
| `ca_tcc` | Có | Có | Có | CA-TCC đầy đủ |
| `ca_scratch` | Không | Có | Có | Ablation mở rộng, không phải cấu hình chính thức |

Tổng ma trận mặc định gồm **6 phương pháp × 3 tỷ lệ nhãn = 18 cấu hình**. Khi chạy 3 seed sẽ có 54 kết quả cuối; checkpoint self-supervised được dùng lại theo từng seed để tránh huấn luyện thừa.

## Chạy dự án từ đầu

Các lệnh dưới đây giả định đang dùng PowerShell trên Windows. Nếu dùng Bash/Linux/Colab, thay dấu xuống dòng `` ` `` bằng `\`.

Mở đúng thư mục dự án:

```powershell
cd D:\timesseries\CKHoanThien\CA-TCC-SpokenArabicDigits-Complete
```

Cài thư viện. Nên dùng `python -m pip` vì một số máy Windows không nhận trực tiếp lệnh `pip`:

```powershell
python -m pip install -r requirements.txt
python -m pip install notebook nbconvert ipykernel
```

Kiểm tra code nhanh, không tải dữ liệu:

```powershell
python scripts\smoke_test.py
```

Kết quả mong đợi:

```text
SMOKE TEST PASSED
logits: (8, 10) features: (8, 128, 14)
```

Smoke test kiểm tra shape encoder, Temporal Contrasting, NT-Xent, SupCon, lan truyền ngược, augmentation MFCC và dataloader `.pt`.

## Tạo dữ liệu lần đầu

Nếu `data/SpokenArabicDigits/` chưa có `train.pt`, `val.pt`, `test.pt`, cần chạy notebook tiền xử lý. Không gõ trực tiếp đường dẫn `.ipynb` trong PowerShell vì như vậy notebook không được execute.

Cách dễ nhất là mở notebook trong VS Code/Jupyter rồi bấm `Run All`:

```text
notebooks/01_EDA_Preprocessing_SpokenArabicDigits.ipynb
```

Hoặc chạy notebook bằng terminal:

```powershell
python -m jupyter nbconvert `
  --to notebook `
  --execute notebooks\01_EDA_Preprocessing_SpokenArabicDigits.ipynb `
  --inplace `
  --ExecutePreprocessor.timeout=-1
```

Khi hoàn thành, kiểm tra:

```powershell
Get-ChildItem data\SpokenArabicDigits
```

Các file dữ liệu nền cần có:

```text
train.pt
val.pt
test.pt
train_1perc.pt
train_5perc.pt
train_10perc.pt
train_50perc.pt
train_75perc.pt
scaler_stats.npz
split_metadata.csv
preprocessing_manifest.json
```

Nếu các file này đã tồn tại, có thể bỏ qua notebook và chạy thẳng phần huấn luyện.

## Chạy huấn luyện trên terminal

Lệnh dưới đây chạy foreground, tức là terminal sẽ hiện từng epoch, loss, accuracy, Macro-F1 và JSON metric cuối mỗi cấu hình.

Chạy gọn để kiểm tra một seed:

```powershell
python experiments\run_experiments.py `
  --methods supervised ts_tcc ca_tcc `
  --label_ratios 1 5 10 `
  --seeds 0 `
  --device auto
```

Chạy so sánh chính TS-TCC và CA-TCC:

```powershell
python experiments\run_experiments.py `
  --methods ts_tcc ca_tcc `
  --label_ratios 1 5 10 `
  --seeds 0 `
  --device auto
```

Chạy toàn bộ baseline và ablation phục vụ báo cáo:

```powershell
python experiments\run_experiments.py `
  --methods supervised ts_tcc ts_pl ts_supcon ca_tcc ca_scratch `
  --label_ratios 1 5 10 `
  --seeds 0 1 2 `
  --device auto
```

Mặc định mỗi stage dùng tối đa 40 epoch và có early stopping ở các stage supervised/linear. Full 3 seed có thể lâu vì gồm nhiều bước:

```text
self_supervised pretrain
fine-tune
generate pseudo-label
SupCon
linear evaluation
test evaluation
```

Trong stage `self_supervised`, terminal chỉ hiện `self_supervised loss`. Các chỉ số `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `weighted_f1`, `cohen_kappa` xuất hiện ở các stage classifier và trong JSON cuối mỗi cấu hình.

## Tiếp tục sau khi tắt máy hoặc bị ngắt

Nếu đã chạy trước đó và máy bị tắt/ngắt giữa chừng, mở lại PowerShell, vào đúng thư mục dự án rồi chạy lại đúng lệnh cũ. Script tự kiểm tra output đã có và bỏ qua stage hoàn tất.

Ví dụ tiếp tục full matrix:

```powershell
cd D:\timesseries\CKHoanThien\CA-TCC-SpokenArabicDigits-Complete
python experiments\run_experiments.py `
  --methods supervised ts_tcc ts_pl ts_supcon ca_tcc ca_scratch `
  --label_ratios 1 5 10 `
  --seeds 0 1 2 `
  --device auto
```

Chạy lại từ đầu, bỏ qua cơ chế resume:

```powershell
python experiments\run_experiments.py ... --force
```

Nếu muốn dọn sạch kết quả thí nghiệm để chạy lại chủ động:

```powershell
Remove-Item experiments_logs -Recurse -Force
Remove-Item data\SpokenArabicDigits\pseudo_* -Force
```

Không xoá các file nền như `train.pt`, `val.pt`, `test.pt`, `train_1perc.pt`, `train_5perc.pt`, `train_10perc.pt`, `scaler_stats.npz`, `split_metadata.csv`, `preprocessing_manifest.json`.

## Chạy nhanh để kiểm tra

Không thực thi huấn luyện, chỉ in ra các stage sẽ chạy:

```powershell
python experiments\run_experiments.py `
  --methods supervised ts_tcc ts_pl ts_supcon ca_tcc ca_scratch `
  --label_ratios 1 5 10 `
  --seeds 0 `
  --dry_run
```

Giới hạn một epoch:

```powershell
python experiments\run_experiments.py `
  --methods supervised ts_tcc ca_tcc `
  --label_ratios 1 5 10 `
  --seeds 0 `
  --epochs 1 `
  --device auto
```

## Chạy CPU

Nên dùng GPU cho thí nghiệm thật. Khi cần chạy CPU:

```powershell
python experiments\run_experiments.py `
  --methods ts_tcc ca_tcc `
  --label_ratios 1 5 10 `
  --seeds 0 `
  --device cpu `
  --cpu_threads 4
```

## Pseudo-label confidence

Mặc định `--confidence_threshold 0.0` giữ toàn bộ pseudo-label, gần với quy trình gốc. Có thể thí nghiệm thêm:

```powershell
python experiments\run_experiments.py `
  --methods ca_tcc `
  --label_ratios 1 5 10 `
  --seeds 0 `
  --device auto `
  --confidence_threshold 0.7
```

Nhãn thật ẩn của benchmark chỉ được dùng để tạo các chỉ số có hậu tố `audit_only`; chúng không được đưa vào huấn luyện ngoài phần 1%, 5% hoặc 10% đã chọn.

## Tổng hợp kết quả

Sau khi huấn luyện xong, tổng hợp toàn bộ `metrics.json` thành bảng CSV và biểu đồ:

```powershell
python experiments\aggregate_results.py `
  --experiment SpokenArabicDigits_matrix `
  --output results
```

Các file nên xem trước:

```text
results/all_runs.csv
results/summary_by_method_and_ratio.csv
results/macro_f1_pivot.csv
results/macro_f1_comparison.png
```

Sau đó có thể mở notebook phân tích:

```text
notebooks/02_Experiment_Result_Analysis.ipynb
```

## File kết quả được lưu ở đâu

Khi chạy huấn luyện, repo tạo `experiments_logs/`. Mỗi stage có một thư mục riêng và thường chứa:

- `metrics.json`, `metrics.csv`;
- `classification_report.csv`;
- `confusion_matrix.npy`;
- `history.csv`;
- `run.log`;
- `saved_models/ckp_best.pt`, `saved_models/ckp_last.pt`.

Các method có pseudo-label như `ts_pl`, `ca_tcc`, `ca_scratch` còn sinh thêm file trong `data/SpokenArabicDigits/`:

- `pseudo_*.pt`;
- `pseudo_*.json`;
- `pseudo_*_audit.csv`.

Đây là dữ liệu trung gian hợp lệ của thí nghiệm. Nếu chỉ muốn phân tích kết quả, xem `results/` và `experiments_logs/**/metrics.json`. Nếu muốn dọn để chạy lại từ đầu, có thể xoá `experiments_logs/` và `data/SpokenArabicDigits/pseudo_*`, nhưng giữ nguyên các file `train*.pt`, `val.pt`, `test.pt` và manifest tiền xử lý.

Các đầu ra tổng hợp quan trọng:

- `orchestration_log.csv`;
- bảng mean/std theo method và tỷ lệ nhãn;
- biểu đồ Macro-F1;
- thời gian pipeline;
- mức cải thiện CA-TCC so với TS-TCC.

## Nguồn và phạm vi thay đổi

Xem:

- `original_source/SOURCE_REFERENCE.md`
- `VALIDATION_REPORT.md`

Lõi được giữ theo thiết kế tác giả gồm encoder Conv1D ba khối, Transformer context encoder, Temporal Contrasting, NT-Xent, pseudo-label và supervised contrastive learning. Các thay đổi chủ yếu là dữ liệu MFCC, augmentation phù hợp lời nói, tham số hóa 1%–5%–10%, ablation và hệ thống lưu kết quả.
