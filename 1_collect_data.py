import cv2
import mediapipe as mp
import csv
import os

# 1. Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

filename = 'gesture_data.csv'

# 2. Tạo file CSV và ghi dòng tiêu đề (Header) nếu file chưa tồn tại
if not os.path.exists(filename):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']] + ['label']
        writer.writerow(header)

# 3. Mở Webcam
cap = cv2.VideoCapture(0)
print("=== TOOL THU THẬP TỌA ĐỘ CỬ CHỈ (BẢN NÂNG CẤP 5 CỬ CHỈ) ===")
print("Hướng dẫn: Đưa tay lên trước camera và làm cử chỉ.")
print("- Nhấn giữ phím '0': XÒE CẢ BÀN TAY (STOP)")
print("- Nhấn giữ phím '1': NGÓN TRỎ LÊN (FORWARD)")
print("- Nhấn giữ phím '2': NGÓN CÁI SANG TRÁI (TURN LEFT)")
print("- Nhấn giữ phím '3': NGÓN CÁI SANG PHẢI (TURN RIGHT)")
print("- Nhấn giữ phím '4': NGÓN CÁI XUỐNG DƯỚI (BACKWARD)")
print("- Nhấn phím 'q' để THOÁT.")
print("================================================")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể kết nối với Webcam!")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    key = cv2.waitKey(1) & 0xFF

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Cập nhật điều kiện từ '0'-'3' thành '0'-'4'
            if ord('0') <= key <= ord('4'):
                label = chr(key)
                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                row.append(label)
                with open(filename, mode='a', newline='') as f:
                    csv.writer(f).writerow(row)
                print(f"Đã ghi data cho cử chỉ nhãn: {label}")

    cv2.imshow("Thu thap du lieu cu chi", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()