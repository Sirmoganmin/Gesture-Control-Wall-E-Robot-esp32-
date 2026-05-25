# Đồ án Học máy: Nhận dạng cử chỉ tay điều khiển Robot
Đề tài: Hệ thống nhận diện cử chỉ bằng MediaPipe và Machine Learning (KNN) để điều khiển robot 4 bánh (Tiến, Lùi, Trái, Phải, Dừng).

## Cài đặt môi trường
Mở Terminal và chạy lệnh sau để cài đặt các thư viện cần thiết:
pip install pandas scikit-learn mediapipe opencv-python matplotlib

## Hướng dẫn chạy code
- Bước 1: Chạy file `1_collect_data.py` (Nếu muốn tự thu thập lại dữ liệu cử chỉ của riêng bạn).
- Bước 2: Chạy file `2_train_model.py` để huấn luyện mô hình AI và sinh ra file `gesture_model.pkl`.
- Bước 3: Chạy file `3_run_simulation.py` để mở Bảng điều khiển ảo, giơ tay trước camera để test mô hình.


## 🚀 Giai đoạn tiếp theo: Tích hợp phần cứng (ESP32 & Robot Wall-E)
Đồ án đang chuẩn bị bước sang Giai đoạn 2 để biến tín hiệu ảo thành chuyển động vật lý:
- Hệ thống Python sẽ được bổ sung giao thức Serial để truyền các mã lệnh (0, 1, 2, 3, 4) qua cáp/Wi-Fi xuống mạch Vi điều khiển ESP32.
- ESP32 sẽ đóng vai trò trung tâm xử lý dưới hạ tầng, tiếp nhận tín hiệu và kích hoạt mạch công suất (Module L298N) để vận hành 4 động cơ của robot Wall-E sao cho khớp thời gian thực với cử chỉ tay của người điều khiển.
