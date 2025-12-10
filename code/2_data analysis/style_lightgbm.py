## 기본 라이브러리
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import early_stopping, log_evaluation
from sklearn.preprocessing import MultiLabelBinarizer
import ast
from sklearn.model_selection import RandomizedSearchCV

# 1) 데이터 로드
#    - train/val 데이터를 불러오고 행 방향으로 병합
data = pd.read_csv(
    "C:/Users/Admin/Desktop/PROJ/data/02_cleaned/clothes/0.top_bottom_only.csv",
    encoding="utf-8-sig",
)
data_val = pd.read_csv(
    "C:/Users/Admin/Desktop/PROJ/data/02_cleaned/clothes_val/0.top_bottom_only.csv",
    encoding="utf-8-sig",
)
data = pd.concat([data, data_val], axis=0).reset_index(drop=True)

# 2) 불필요한 컬럼 제거
#    - 모델 학습에 필요 없는 이미지/좌표/세부 디테일 컬럼 제거
data = data.drop(
    columns=[
        "식별자",
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
# # 2) 리스트 변환 함수
# #    - 문자열, 쉼표/구분자, Python 리스트 형태 등 다양한 입력을
# #      통일된 리스트 형태로 변환
# # -------------------------
# def to_list(x):
#     if pd.isna(x):
#         return ["nan"]
#     if isinstance(x, list):
#         return [str(v).strip() for v in x if (v is not None and str(v).strip()!="")]
#     if isinstance(x, str):
#         s = x.strip()
#         # a) Python list string → literal_eval
#         try:
#             val = ast.literal_eval(s)
#             if isinstance(val, list):
#                 return [str(v).strip() for v in val if (v is not None and str(v).strip()!="")]
#         except Exception:
#             pass
#         # b) 쉼표로 분리된 multi-label 문자열 처리
#         if "," in s:
#             parts = [p.strip() for p in s.split(",") if p.strip()!=""]
#             return parts
#         # c) 기타 구분자(| / ;) 처리
#         for sep in ["|", "/", ";"]:
#             if sep in s:
#                 parts = [p.strip() for p in s.split(sep) if p.strip()!=""]
#                 return parts
#         # d) 단일 문자열
#         return [s]
#     return [str(x).strip()]

# # -------------------------
# # 3) MultiLabelBinarizer 적용
# #    - 상의_소재 / 하의_소재 컬럼을 multi-hot encoding으로 확장
# # -------------------------
# for col in ['상의_소재', '하의_소재']:
#     data[col] = data[col].apply(to_list)

#     mlb = MultiLabelBinarizer()
#     expanded = pd.DataFrame(
#         mlb.fit_transform(data[col]),
#         columns=[f"{col}_{cls}" for cls in mlb.classes_],
#         index=data.index
#     )

#     # 원본 컬럼 제거 후 인코딩된 다차원 컬럼 병합
#     data = pd.concat([data.drop(columns=[col]), expanded], axis=1)

# 3) X, y 분리
X = data.drop("스타일", axis=1)
y = data["스타일"]

# 4) 타겟(Label) 인코딩
le = LabelEncoder()
y_encoded = le.fit_transform(y)
label_mapping = dict(enumerate(le.classes_))


# 6) Train/Test 분리
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


# 9) Class Weight 계산
#    - 불균형 데이터 대응용 가중치 자동 계산
counter = Counter(y_train)
max_count = max(counter.values())
class_weights = {cls: (max_count / cnt) ** 0.5 for cls, cnt in counter.items()}
cb_weights = [class_weights[i] for i in sorted(class_weights.keys())]

# 10) LightGBM 모델 정의
#     - multiclass 분류, early-stopping, 로그 설정 포함
lgb_param_grid = {
    "num_leaves": [31, 50, 70],
    "max_depth": [-1, 8, 12],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [200, 400, 600],
    "min_child_samples": [10, 20, 30],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}

model_lgb = RandomizedSearchCV(
    estimator=LGBMClassifier(
        objective="multiclass",
        num_class=len(label_mapping),
        class_weight=class_weights,
        random_state=42,
        n_jobs=1
    ),
    param_distributions=lgb_param_grid,
    n_iter=20,
    scoring="accuracy",
    cv=3,
    verbose=2,
    random_state=42
)

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

print("Best LGB Params:", model_lgb.best_params_)
print("Best LGB Score:", model_lgb.best_score_)

