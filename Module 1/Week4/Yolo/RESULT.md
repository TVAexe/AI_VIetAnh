# Kết quả đánh giá mô hình YOLOv10n

- **Môi trường:**  
  - Ultralytics YOLOv8.1.34  
  - Python 3.9.23  
  - Torch 2.5.1 + CUDA 12.1  
  - GPU: Quadro P2000 (4GB)

- **Kiến trúc mô hình:**  
  - 285 layers, 2.7 triệu tham số, 8.2 GFLOPs

- **Tập kiểm thử:**  
  - 109 ảnh, 320 đối tượng

## Kết quả từng lớp

| Lớp     | Precision | Recall | mAP50 | mAP50-95 |
|---------|-----------|--------|-------|----------|
| **all**     | 0.83      | 0.785  | 0.87  | 0.428    |
| head    | 0.963     | 0.688  | 0.893 | 0.389    |
| helmet  | 0.806     | 0.92   | 0.925 | 0.456    |
| person  | 0.72      | 0.746  | 0.792 | 0.44     |

- **Tốc độ:**  
  - 2.3ms tiền xử lý/ảnh  
  - 12.8ms suy luận/ảnh  
  - 1.8ms hậu xử lý/ảnh