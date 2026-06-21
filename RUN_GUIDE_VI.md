# HƯỚNG DẪN CHẠY DỰ ÁN

## 1. Giải nén và mở đúng thư mục

Giải nén `CA-TCC-SpokenArabicDigits-Complete.zip`. Mọi lệnh dưới đây cần chạy từ thư mục:

```text
CA-TCC-SpokenArabicDigits-Complete/
```

Trên Google Colab, tải file ZIP lên, giải nén rồi chuyển thư mục làm việc:

```python
!unzip -q CA-TCC-SpokenArabicDigits-Complete.zip -d /content/
%cd /content/CA-TCC-SpokenArabicDigits-Complete
```

## 2. Cài thư viện

```bash
pip install -r requirements.txt
```

## 3. Kiểm tra code trước khi tải dữ liệu

```bash
python scripts/smoke_test.py
```

Kết quả mong đợi:

```text
SMOKE TEST PASSED
logits: (8, 10) features: (8, 128, 14)
```

## 4. Chạy notebook EDA và tiền xử lý

Mở:

```text
notebooks/01_EDA_Preprocessing_SpokenArabicDigits.ipynb
```

Chạy lần lượt từ cell đầu đến cell cuối. Notebook tự tải dữ liệu UCI; không cần tải thủ công trong điều kiện mạng cho phép.

Đầu ra cần có:

```text
data/SpokenArabicDigits/
├── train.pt
├── val.pt
├── test.pt
├── train_1perc.pt
├── train_5perc.pt
├── train_10perc.pt
├── train_50perc.pt
├── train_75perc.pt
├── scaler_stats.npz
├── split_metadata.csv
└── preprocessing_manifest.json
```

## 5. Chạy thử một seed

```bash
python experiments/run_experiments.py \
  --methods supervised ts_tcc ca_tcc \
  --label_ratios 1 5 10 \
  --seeds 0 \
  --device auto
```

Đây là bước nên chạy trước để kiểm tra toàn bộ pipeline trên máy của bạn.

## 6. Chạy ma trận phục vụ báo cáo

```bash
python experiments/run_experiments.py \
  --methods supervised ts_tcc ts_pl ts_supcon ca_tcc ca_scratch \
  --label_ratios 1 5 10 \
  --seeds 0 1 2 \
  --device auto
```

Ý nghĩa:

- `supervised`: baseline không tiền huấn luyện;
- `ts_tcc`: phương pháp TS-TCC riêng;
- `ts_pl`: TS-TCC cộng pseudo-label nhưng không SupCon;
- `ts_supcon`: TS-TCC cộng SupCon chỉ trên nhãn thật;
- `ca_tcc`: TS-TCC cộng pseudo-label và SupCon;
- `ca_scratch`: phần CA khởi đầu từ supervised, không dùng TS-TCC.

## 7. Tiếp tục khi bị ngắt

Chạy lại đúng lệnh cũ. Script tự kiểm tra checkpoint/metric và bỏ qua giai đoạn đã hoàn tất.

Chạy lại từ đầu:

```bash
python experiments/run_experiments.py ... --force
```

## 8. Chạy nhanh để kiểm tra lệnh

Không thực thi huấn luyện:

```bash
python experiments/run_experiments.py \
  --methods supervised ts_tcc ts_pl ts_supcon ca_tcc ca_scratch \
  --label_ratios 1 5 10 \
  --seeds 0 \
  --dry_run
```

Giới hạn một epoch:

```bash
python experiments/run_experiments.py \
  --methods supervised ts_tcc ca_tcc \
  --label_ratios 1 5 10 \
  --seeds 0 \
  --epochs 1 \
  --device auto
```

## 9. Tổng hợp kết quả

```bash
python experiments/aggregate_results.py \
  --experiment SpokenArabicDigits_matrix \
  --output results
```

Mở tiếp:

```text
notebooks/02_Experiment_Result_Analysis.ipynb
```

## 10. Các so sánh nên trình bày trong báo cáo

```text
supervised ↔ ts_tcc
```

Đo lợi ích của self-supervised pretraining.

```text
ts_tcc ↔ ts_pl
```

Đo đóng góp của pseudo-label.

```text
ts_tcc ↔ ts_supcon
```

Đo đóng góp của SupCon khi chỉ dùng nhãn thật.

```text
ts_tcc ↔ ca_tcc
```

So sánh chính giữa TS-TCC và CA-TCC đầy đủ.

```text
ca_scratch ↔ ca_tcc
```

Kiểm tra phần CA có cần nền biểu diễn TS-TCC hay không.

## 11. Lưu ý

- Nên dùng Colab GPU hoặc GPU cục bộ.
- Chạy seed 0 trước, sau đó mới mở rộng sang 0, 1, 2.
- `ca_scratch` là thí nghiệm ablation do dự án bổ sung, không phải cấu hình chính thức của bài báo.
- Không dùng kết quả smoke test hoặc dữ liệu giả làm kết quả báo cáo.
