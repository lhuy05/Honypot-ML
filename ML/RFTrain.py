import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Đọc dataset đã chuẩn hóa
df = pd.read_csv("train_featuresnew1.csv")

# Tách X và y
X = df.drop("label", axis=1)
y = df["label"]

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

# Đánh giá
print(X.iloc[:20].to_string())
pred = model.predict(X_test)
print("\n=== BÁO CÁO PHÂN LOẠI RANDOM FOREST ===")
print(classification_report(y_test, pred))

# Feature importance
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

print(
    importance.sort_values(
        "importance",
        ascending=False
    )
)

# Lưu model
joblib.dump(model, "attack_model.pkl")

print("Saved attack_model.pkl")
sample_idx = 19
print(f"\n--- Test sample {sample_idx} ---")
print(X.iloc[sample_idx].to_string())
print("Predicted Class:", model.predict(X.iloc[[sample_idx]]))
print("Probabilities:", model.predict_proba(X.iloc[[sample_idx]]))
