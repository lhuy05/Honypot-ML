import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# ==========================================
# 1. ĐỌC VÀ CHUẨN BỊ DỮ LIỆU
# ==========================================
df = pd.read_csv("train_featuresnew1.csv")

# Tách X và y (X tự động nhận đủ 16 đặc trưng mới)
X = df.drop("label", axis=1).astype(np.float32)
y_raw = df["label"]

# Mã hóa nhãn từ chuỗi -> số
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 2. KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH XGBOOST
# ==========================================
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric="mlogloss"
)

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Lưu danh sách nhãn chuỗi gốc vào thuộc tính tùy biến
model.custom_classes_ = label_encoder.classes_

# ==========================================
# 3. ĐÁNH GIÁ MÔ HÌNH
# ==========================================
print(X.iloc[:20].to_string())
pred = model.predict(X_test)

y_test_labels = label_encoder.inverse_transform(y_test)
pred_labels = label_encoder.inverse_transform(pred)

print("\n=== BÁO CÁO PHÂN LOẠI XGBOOST ===")
print(classification_report(y_test_labels, pred_labels))

# ==========================================
# 4. ĐỘ QUAN TRỌNG CỦA ĐẶC TRƯNG (FEATURE IMPORTANCE)
# ==========================================
importance = pd.DataFrame(
    {"feature": X.columns, "importance": model.feature_importances_}
)
print("\n=== ĐỘ QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG ===")
print(importance.sort_values("importance", ascending=False))

# ==========================================
# 5. LƯU MÔ HÌNH VÀ BỘ MÃ HÓA NHÃN
# ==========================================
joblib.dump(model, "attack_modelxgnew1.pkl")
joblib.dump(label_encoder, "label_encoderxgnew1.pkl")
print("\n💾 Đã lưu attack_modelxg.pkl và label_encoderxg.pkl")

# ==========================================
# 6. KIỂM TRA THỬ 1 MẪU DỮ LIỆU (SAMPLE TEST)
# ==========================================
sample_idx = 19
print(f"\n--- Test sample {sample_idx} ---")
print(X.iloc[sample_idx].to_string())

sample_data = X.iloc[[sample_idx]]
pred_num = model.predict(sample_data)
pred_class = label_encoder.inverse_transform(pred_num)

print("Predicted Class:", pred_class[0])
print("Probabilities:")

proba_scores = model.predict_proba(sample_data)[0]
for cls, prob in zip(model.custom_classes_, proba_scores):
    print(f"  ↳ {cls}: {prob:.4f}")