import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 1. Đọc dữ liệu CSV
print("Đang tải dữ liệu từ gesture_data.csv...")
try:
    df = pd.read_csv('gesture_data.csv')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file gesture_data.csv.")
    exit()

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 2. Chia tập Train/Test (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Lên danh sách các thuật toán
models = {
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine (SVM)": SVC(kernel='linear', random_state=42)
}

best_acc = 0
best_model_name = ""
best_model = None

print("\n--- BẢNG XẾP HẠNG ĐỘ CHÍNH XÁC ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"| {name:<30} | {acc * 100:>6.2f}% |")

    if acc > best_acc:
        best_acc = acc
        best_model_name = name
        best_model = model

print("-" * 40)
print(f"🏆 MÔ HÌNH TỐT NHẤT: {best_model_name}")

# --- PHẦN MỚI THÊM: IN BÁO CÁO CHI TIẾT ĐỂ BÊ VÀO ĐỒ ÁN ---
print(f"\n================ BÁO CÁO CHI TIẾT CỦA MÔ HÌNH {best_model_name} ================")
y_pred_best = best_model.predict(X_test)
target_names = ['0 (Stop)', '1 (Tien)', '2 (Trai)', '3 (Phai)', '4 (Lui)']

# Lệnh này sẽ in ra Precision, Recall, F1-Score
print(classification_report(y_test, y_pred_best, target_names=target_names))
print("========================================================================")

# 5. Lưu mô hình
with open('gesture_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("\n[OK] Đã lưu mô hình tốt nhất vào file 'gesture_model.pkl'.")

# 6. Vẽ Ma trận nhầm lẫn
print("Đang hiển thị biểu đồ Ma trận nhầm lẫn...")
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Ma tran nham lan - {best_model_name}")
plt.show()
