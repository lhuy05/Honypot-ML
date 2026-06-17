import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree thay cho Random Forest

# ==========================================
# 1. ĐỌC VÀ CHUẨN BỊ DỮ LIỆU
# ==========================================
# Đọc dataset đã chuẩn hóa
df = pd.read_csv("train_featuresnew1.csv")

# Tách X và y
X = df.drop("label", axis=1)
y = df["label"]

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 2. KHỞI TẠO VÀ HUẤN LUYỆN DECISION TREE
# ==========================================
# Train bằng DecisionTreeClassifier
model = DecisionTreeClassifier(
    criterion="gini",  # Thước đo độ tinh khiết (hoặc dùng "entropy")
    max_depth=5,  # Để cây phát triển tự nhiên (hoặc giới hạn lại nếu bị overfit)
    random_state=42,
)

model.fit(X_train, y_train)

# ==========================================
# 3. ĐÁNH GIÁ MÔ HÌNH
# ==========================================
print(X.iloc[:20].to_string())
pred = model.predict(X_test)

print("\n=== BÁO CÁO PHÂN LOẠI (DECISION TREE) ===")
print(classification_report(y_test, pred))

# ==========================================
# 4. ĐỘ QUAN TRỌNG CỦA ĐẶC TRƯNG (FEATURE IMPORTANCE)
# ==========================================
importance = pd.DataFrame(
    {"feature": X.columns, "importance": model.feature_importances_}
)

print("\n=== ĐỘ QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG ===")
print(importance.sort_values("importance", ascending=False))

# ==========================================
# 5. LƯU MÔ HÌNH
# ==========================================
joblib.dump(model, "attack_model.pkl")
print("\nSaved attack_model.pkl")

# ==========================================
# 6. KIỂM TRA THỬ MẪU DỮ LIỆU (SAMPLE TEST)
# ==========================================
sample_idx = 19
print(f"\n--- Test sample {sample_idx} ---")
print(X.iloc[sample_idx].to_string())
print("Predicted Class:", model.predict(X.iloc[[sample_idx]]))
print("Probabilities:", model.predict_proba(X.iloc[[sample_idx]]))