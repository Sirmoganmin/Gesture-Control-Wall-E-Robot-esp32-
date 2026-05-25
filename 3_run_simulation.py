import cv2
import mediapipe as mp
import pickle

# 1. Tải mô hình AI đã train
try:
    with open('gesture_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Đã nạp bộ não AI thành công! Đang mở Webcam...")
except FileNotFoundError:
    print("LỖI: Chưa có file 'gesture_model.pkl'. Bạn cần chạy file Train trước nhé!")
    exit()

# 2. Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# 3. Bảng thiết lập Giao diện Mô phỏng
robot_states = {
    '0': ["ROBOT DUNG (STOP)", (0, 0, 255)],  # Đỏ
    '1': ["ROBOT TIEN LEN (FORWARD)", (0, 255, 0)],  # Xanh lá
    '2': ["RE TRAI (TURN LEFT)", (0, 255, 255)],  # Vàng
    '3': ["RE PHAI (TURN RIGHT)", (0, 255, 255)],  # Vàng
    '4': ["ROBOT LUI (BACKWARD)", (255, 0, 255)]  # Tím
}

# 4. Mở Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể kết nối camera!")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    current_state = "KHONG THAY TAY"
    box_color = (128, 128, 128)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            # ĐÃ FIX LỖI Ở ĐÂY: Thêm str() để ép kiểu dữ liệu về chuỗi
            prediction = str(model.predict([row])[0])

            if prediction in robot_states:
                current_state = robot_states[prediction][0]
                box_color = robot_states[prediction][1]

    # --- VẼ BẢNG ĐIỀU KHIỂN ---
    cv2.rectangle(frame, (10, 10), (550, 80), box_color, cv2.FILLED)
    cv2.rectangle(frame, (10, 10), (550, 80), (0, 0, 0), 2)

    cv2.putText(frame, f"Trang thai: {current_state}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(frame, f"Trang thai: {current_state}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Dashboard Mo Phong Robot", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()