import os
import time
import joblib
import numpy as np
import pandas as pd

# ==========================================
# 1. CẤU HÌNH
# ==========================================

INPUT_FEATURES_FILE = "realtime_features_outputnew1.csv"

print("=== [SYSTEM] Đang khởi động Random Forest Realtime Detector... ===")

try:
    model = joblib.load("attack_model.pkl")
    # Lấy danh sách class từ model RF
    classes = model.classes_

    print("=== [SYSTEM] Load Random Forest thành công ===")
    print(f"=== [SYSTEM] Classes: {list(classes)} ===")
    print("=== [SYSTEM] Đang đợi request mới... ===\n")

except Exception as e:
    print(f"=== [ERROR] Không thể load model: {e} ===")
    exit()

# ==========================================
# 2. DANH SÁCH FEATURE
# PHẢI TRÙNG THỨ TỰ VỚI train_featuresnew1.csv
# ==========================================

feature_headers = [
    "method",
    "path_length",
    "query_length",
    "body_length",
    "slash_count",
    "dot_count",
    "path_special_chars",
    "query_special_chars",
    "body_special_chars",
    "digit_ratio",
    "percent_encoding_count",
    "has_sql_keyword",
    "sqli_pattern_count",
    "has_xss_keyword",
    "has_path_traversal",
    "has_cmd_keyword",
    "html_bracket_count",
    "quote_count",
    "js_bracket_count"
]

# ==========================================
# 3. DỰ ĐOÁN 1 REQUEST
# ==========================================

def predict_single_request(features_dict):
    X_single = pd.DataFrame([features_dict])
    X_single = X_single[feature_headers]

    prediction = model.predict(X_single)[0]
    probabilities = model.predict_proba(X_single)[0]

    return prediction, probabilities

# ==========================================
# 4. HIỂN THỊ XÁC SUẤT
# ==========================================

def print_probabilities(probabilities):
    print("\n===== RAW PROBABILITIES =====")
    for cls, prob in zip(classes, probabilities):
        print(f"{cls}: {prob:.4f}")

# ==========================================
# 5. ĐỌC FILE REALTIME LIÊN TỤC
# ==========================================

def watch_and_display():
    while not os.path.exists(INPUT_FEATURES_FILE):
        print(f"⚠️ Đang đợi file đặc trưng: {INPUT_FEATURES_FILE}")
        time.sleep(1)

    with open(INPUT_FEATURES_FILE, "r", encoding="utf-8") as f_in:
        # Bỏ dữ liệu cũ, chỉ đọc dữ liệu mới phát sinh
        f_in.seek(0, os.SEEK_END)

        while True:
            line = f_in.readline()

            if not line:
                time.sleep(0.05)
                continue

            line = line.strip()
            if not line:
                continue

            try:
                parts = [p.strip() for p in line.split(",")]

                # Bỏ qua dòng header nếu có
                if parts[0].lower() in ["method", "method_map"]:
                    continue

                if len(parts) < len(feature_headers):
                    continue

                features_dict = {}
                for idx, feature_name in enumerate(feature_headers):
                    features_dict[feature_name] = float(parts[idx])

                prediction, probabilities = predict_single_request(features_dict)
                confidence = np.max(probabilities)

                print_probabilities(probabilities)

                if prediction == "Normal":
                    print(f"\n🟢 [REQUEST SẠCH]")
                    print(f"   ↳ Kết quả: {prediction}")
                    print(f"   ↳ Độ tin cậy: {confidence:.4f}")
                else:
                    print(f"\n🚨 [CẢNH BÁO TẤN CÔNG]")
                    print(f"   ↳ Loại tấn công: {prediction}")
                    print(f"   ↳ Độ tin cậy: {confidence:.4f}")
                    print(f"   ↳ Chi tiết: {dict(zip(classes, np.round(probabilities, 4)))}")

                print("-" * 60)

            except Exception as e:
                print(f"❌ Lỗi xử lý request: {e}")

# ==========================================
# 6. MAIN
# ==========================================

if __name__ == "__main__":
    try:
        watch_and_display()
    except KeyboardInterrupt:
        print("\n👋 Đã dừng Random Forest Detector")