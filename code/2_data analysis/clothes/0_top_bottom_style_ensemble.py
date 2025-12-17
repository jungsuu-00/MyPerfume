## 기본 라이브러리
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.model_selection import RandomizedSearchCV

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import category_encoders as ce
import ast
import joblib

# 1) 데이터 로드
data = pd.read_csv(
    "data/02_cleaned/clothes/0.top_bottom_only.csv",
    encoding="utf-8-sig",
)
data_val = pd.read_csv(
    "data/02_cleaned/clothes_val/0.top_bottom_only.csv",
    encoding="utf-8-sig",
)
data = pd.concat([data, data_val], axis=0).reset_index(drop=True)

if "서브스타일" in data.columns:
    data["상의_서브스타일"] = data["서브스타일"]
    data["하의_서브스타일"] = data["서브스타일"]

# 2) 불필요한 컬럼 제거
data = data.drop(
    columns=[
        "식별자",
        "서브스타일",
        "렉트좌표_상의",
        "렉트좌표_하의",
        "파일명",
        "상의_기장",
        "상의_디테일",
        "하의_디테일",
        "상의_넥라인",
        "하의_프린트",
    ],
    axis=1,
)

# 상의 카테고리가 '탑', '브라탑'인 행 제거
data["상의_카테고리"] = data["상의_카테고리"].replace({"브라탑": "탑"})

# # -------------------------
# # 리스트 변환 함수
# # -------------------------
# def to_list(x):
#     if pd.isna(x):
#         return ["nan"]
#     if isinstance(x, list):
#         return [str(v).strip() for v in x if (v is not None and str(v).strip() != "")]
#     if isinstance(x, str):
#         s = x.strip()
#         try:
#             val = ast.literal_eval(s)
#             if isinstance(val, list):
#                 return [
#                     str(v).strip()
#                     for v in val
#                     if (v is not None and str(v).strip() != "")
#                 ]
#         except:
#             pass
#         if "," in s:
#             return [p.strip() for p in s.split(",") if p.strip() != ""]
#         for sep in ["|", "/", ";"]:
#             if sep in s:
#                 return [p.strip() for p in s.split(sep) if p.strip() != ""]
#         return [s]
#     return [str(x).strip()]


# # -------------------------
# # MultiLabelBinarizer 적용
# # -------------------------
# for col in ["상의_소재", "하의_소재"]:
#     data[col] = data[col].apply(to_list)
#     mlb = MultiLabelBinarizer()
#     expanded = pd.DataFrame(
#         mlb.fit_transform(data[col]),
#         columns=[f"{col}_{cls}" for cls in mlb.classes_],
#         index=data.index,
#     )
#     data = pd.concat([data.drop(columns=[col]), expanded], axis=1)

# -------------------------
# X, y 분리
# -------------------------
X = data.drop("스타일", axis=1)
y = data["스타일"]

# -------------------------
# 타겟 인코딩
# -------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)
label_mapping = dict(enumerate(le.classes_))

# -------------------------
# Train/Test 분리
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -------------------------
# 결측치 mode(각 스타일 그룹별)로 채우기
# -------------------------
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

# -------------------------
#  범주형 인코딩 (LightGBM, RF 용)
# -------------------------
cat_cols = X_train.select_dtypes(include=["object"]).columns
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])

# -------------------------
# Class weight (sqrt 완화)
# -------------------------
counter = Counter(y_train)
max_count = max(counter.values())
class_weights = {cls: (max_count / cnt) ** 0.5 for cls, cnt in counter.items()}
cb_weights = [class_weights[i] for i in sorted(class_weights.keys())]

# -------------------------
# Base Models 정의
# -------------------------
model_lgb = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=-1,
    class_weight=class_weights,
    objective="multiclass",
    num_class=len(label_mapping),
    random_state=42,
    n_jobs=1,
)

# model_rf = RandomForestClassifier(
#     n_estimators=400, class_weight=class_weights, random_state=42, n_jobs=1
# )

model_cb = CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=1,
    bagging_temperature=0.5,
    loss_function="MultiClass",
    class_weights=cb_weights,
    random_seed=42,
    verbose=200,
    thread_count=1,
)


## ===============================
## 13) 개별 모델 학습 및 평가
## ===============================
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


model_lgb = evaluate_model(model_lgb, X_train, y_train, X_test, y_test, "LightGBM")
# model_rf = evaluate_model(model_rf, X_train, y_train, X_test, y_test, "RandomForest")
model_cb = evaluate_model(model_cb, X_train, y_train, X_test, y_test, "CatBoost")

## ===============================
## 14) Soft Voting Ensemble
## ===============================
ensemble = VotingClassifier(
    estimators=[
        ("lgb", model_lgb),
        # ("rf", model_rf),
        ("cb", model_cb),
    ],
    voting="soft",
    n_jobs=-1,
)
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

print("\n" + "=" * 60)
print("### 앙상블 성능 (Soft Voting)")
print("=" * 60)
print("Accuracy :", accuracy_score(y_test, y_pred))
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[label_mapping[i] for i in sorted(label_mapping.keys())],
    )
)

# ## ===============================
# ## 15) Stacking Ensemble (Scikit-Learn)
# ## ===============================
# # 메타 모델 정의
# meta_model = LogisticRegression(
#     C=1.0,
#     max_iter=2000,
#     random_state=42,
#     n_jobs=1
# )

# # StackingClassifier 정의
# # 앞서 정의한 조용한(Silent) 모델들을 estimators로 사용
# stacking_clf = StackingClassifier(
#     estimators=[
#         ('lgb', model_lgb),
#         ('xgb', model_xgb),
#         ('cb', model_cb)
#     ],
#     final_estimator=meta_model,
#     cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
#     stack_method='predict_proba',
#     n_jobs=-1,
#     passthrough=False
# )

# print("\n" + "="*60)
# print("### Stacking Ensemble 학습 및 평가")
# print("="*60)

# # 학습 (이제 로그가 출력되지 않고 잠시 기다리면 결과만 나옵니다)
# stacking_clf.fit(X_train, y_train)

# # 예측
# stack_pred = stacking_clf.predict(X_test)

# # 평가
# print("Stacking Accuracy :", accuracy_score(y_test, stack_pred))
# print(classification_report(
#     y_test, stack_pred,
#     target_names=[label_mapping[i] for i in sorted(label_mapping.keys())]
# ))
joblib.dump(
    ensemble,
    "code/2_data analysis/clothes/0_style_model.pkl",
)
joblib.dump(
    encoder,
    "code/2_data analysis/clothes/0_clothes_encoder.pkl",
)
joblib.dump(
    le,
    "code/2_data analysis/clothes/0_style_label_encoder.pkl",
)
