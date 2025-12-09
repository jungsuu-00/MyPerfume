## 데이터 로드 및 기본 라이브러리
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import ast


# 1) 데이터 로드
data = pd.read_csv(
    "C:/Users/Admin/Desktop/PROJ/data/02_cleaned/clothes/0.top_bottom_only.csv",
    encoding="utf-8-sig",
)
data_val = pd.read_csv(
    "C:/Users/Admin/Desktop/PROJ/data/02_cleaned/clothes_val/0.top_bottom_only.csv",
    encoding="utf-8-sig",
)

# 아래로 합치기 (행 기준)
data = pd.concat([data, data_val], axis=0).reset_index(drop=True)

print(data.shape)


# 2) 불필요한 컬럼 제거
data = data.drop(
    columns=[
        "식별자",
        "렉트좌표_상의",
        "렉트좌표_하의",
        "파일명",
        "상의_기장",
        "상의_디테일",
        "상의_넥라인",
        "하의_디테일",
        "하의_프린트",
    ],
    axis=1,
)

# 3) X, y 분리
X = data.drop("스타일", axis=1)
y = data["스타일"]

# 4) 라벨 인코딩 (타겟만)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
label_mapping = dict(enumerate(le.classes_))  # 역매핑


# 5) 조합 Feature 생성
X["색상_조합"] = X["상의_색상"].astype(str) + "_" + X["하의_색상"].astype(str)
X["핏_조합"] = X["상의_핏"].astype(str) + "_" + X["하의_핏"].astype(str)
# X["소재_조합"] = X["상의_소재"].astype(str) + "_" + X["하의_소재"].astype(str)

# 6) train / test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded
)

print("\nTrain set class distribution:")
train_counter = Counter(y_train)
for k, v in sorted(train_counter.items()):
    print(f"  label {k} ({label_mapping[k]}): {v}")

# 7) y(스타일) 기반 최빈값 계산 (train 기준)
train_df = X_train.copy()
train_df["스타일"] = y_train

group_mode_map = {}

for col in X_train.columns:
    mode_by_style = train_df.groupby("스타일")[col].agg(
        lambda x: x.mode()[0] if not x.mode().empty else "Unknown"
    )
    group_mode_map[col] = mode_by_style


# 8) 최빈값으로 결측 대체
def fill_with_style_mode(row, style, mode_map):
    for col in row.index:
        if pd.isna(row[col]):
            if style in mode_map[col].index:
                row[col] = mode_map[col].loc[style]
            else:
                row[col] = "Unknown"
    return row


# train 적용
X_train_filled = X_train.copy()
for idx in X_train_filled.index:
    style = y_train[list(X_train.index).index(idx)]
    X_train_filled.loc[idx] = fill_with_style_mode(
        X_train_filled.loc[idx], style, group_mode_map
    )

# test 적용 (train 기준 최빈값 사용)
X_test_filled = X_test.copy()
for idx in X_test_filled.index:
    style = y_test[list(X_test.index).index(idx)]
    X_test_filled.loc[idx] = fill_with_style_mode(
        X_test_filled.loc[idx], style, group_mode_map
    )

# 9) object → string 통일
for col in X_train_filled.columns:
    if X_train_filled[col].dtype == "object":
        X_train_filled[col] = X_train_filled[col].astype(str)
        X_test_filled[col] = X_test_filled[col].astype(str)

# 10) categorical feature index 추출
cat_features = [
    i
    for i, col in enumerate(X_train_filled.columns)
    if X_train_filled[col].dtype == "object"
]

print(f"\nCategorical features: {cat_features}")

# 11) class_weights 계산
counter = Counter(y_train)
max_count = max(counter.values())
class_weights = {cls: max_count / cnt for cls, cnt in counter.items()}

# 12) CatBoost 모델
model = CatBoostClassifier(
    iterations=600,
    depth=8,
    learning_rate=0.05,
    eval_metric="TotalF1",
    loss_function="MultiClass",
    class_weights=class_weights,
    l2_leaf_reg=6,
    random_strength=0.8,
    bagging_temperature=1,
    verbose=100,
    random_state=42,
)

# 13) 학습
model.fit(
    X_train_filled,
    y_train,
    eval_set=(X_test_filled, y_test),
    early_stopping_rounds=30,
    cat_features=cat_features,
)

# 14) 평가
y_pred = model.predict(X_test_filled)
y_proba = model.predict_proba(X_test_filled)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {acc:.4f}\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[label_mapping[i] for i in sorted(label_mapping.keys())],
    )
)
