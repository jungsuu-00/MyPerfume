## 데이터 로드 및 기본 라이브러리
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV

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
# 상의 카테고리가 '탑', '브라탑'인 행 제거
data["상의_카테고리"] = data["상의_카테고리"].replace({"브라탑": "탑"})

# 3) X, y 분리
X = data.drop("스타일", axis=1)
y = data["스타일"]

# 4) 라벨 인코딩 (타겟만)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
label_mapping = dict(enumerate(le.classes_))  # 역매핑


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

train_df = X_train.copy()
train_df["스타일"] = y_train

for col in X_train.columns:
    mode_per_style = train_df.groupby("스타일")[col].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
    )

    X_train[col] = [
        mode_per_style[style] if pd.isna(val) else val
        for val, style in zip(X_train[col], y_train)
    ]

    X_test[col] = [
        mode_per_style[style] if pd.isna(val) else val
        for val, style in zip(X_test[col], y_test)
    ]
# -------------------------
# 조합 feature 생성
# -------------------------
for df in [X_train, X_test]:
    df["색상_조합"] = df["상의_색상"] + "_" + df["하의_색상"]
    df["핏_조합"] = df["상의_핏"] + "_" + df["하의_핏"]
    # df["소재_조합"] = df["상의_소재"] + "_" + df["하의_소재"]


# 10) categorical feature index 추출
cat_cols = X_train.select_dtypes(include=["object"]).columns
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])


# 11) class_weights 계산
counter = Counter(y_train)
max_count = max(counter.values())
class_weights = {cls: (max_count / cnt) ** 0.5 for cls, cnt in counter.items()}
cb_weights = [class_weights[i] for i in sorted(class_weights.keys())]

# 12) CatBoost 모델
cb_param_grid = {
    "depth": [6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "iterations": [200, 500, 800],
    "l2_leaf_reg": [1, 3, 5],
    "bagging_temperature": [0.0, 0.5, 1.0],
}

model_cb = RandomizedSearchCV(
    estimator=CatBoostClassifier(
        loss_function="MultiClass",
        class_weights=cb_weights,
        random_seed=42,
        verbose=0
    ),
    param_distributions=cb_param_grid,
    n_iter=20,
    scoring="accuracy",
    cv=3,
    verbose=2,
    random_state=42
)

# 13) 학습
def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("\n" + "=" * 60)
    print(f"### {name} 성능")
    print("=" * 60)
    print("Accuracy :", accuracy_score(y_test, pred))
    print(
        classification_report(
            y_test,
            pred,
            target_names=[label_mapping[i] for i in sorted(label_mapping.keys())],
        )
    )
    return model

model_cb = evaluate_model(model_cb, X_train, y_train, X_test, y_test, "CatBoost")
print("Best CB Params:", model_cb.best_params_)
print("Best CB Score:", model_cb.best_score_)