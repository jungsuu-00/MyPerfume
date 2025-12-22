import pandas as pd
import math
import re
import joblib
from collections import defaultdict

# =========================================================
# 1. 데이터 불러오기
# =========================================================
user_df = pd.read_csv("data/03_results/clothes/user_info.csv")
perfume_df = pd.read_csv("data/03_results/perfume/perfume.csv")
perfume_classification = pd.read_csv("data/03_results/perfume/perfume_classification.csv")
perfume_season = pd.read_csv("data/03_results/perfume/perfume_season.csv")

top_bottom = pd.read_csv("data/03_results/clothes/상의_하의.csv")
onepiece = pd.read_csv("data/03_results/clothes/원피스.csv")

clothes_color = pd.read_csv("data/03_results/clothes/clothes_color.csv")
perfume_color = pd.read_csv("data/03_results/perfume/perfume_color.csv")

# =========================================================
# 2. 사용자 선택
# =========================================================
user_id = 1
user_row = user_df.loc[user_df["사용자_식별자"] == user_id].iloc[0]

# 비선호 향조 처리
dislike_raw = user_df.loc[
    user_df["사용자_식별자"] == user_id, "비선호_향조"
].iloc[0]

if isinstance(dislike_raw, str):
    dislike_accords = [x.strip() for x in dislike_raw.split(",")]
elif isinstance(dislike_raw, list):
    dislike_accords = dislike_raw
else:
    dislike_accords = []

mask_exclude = perfume_df[["mainaccord1", "mainaccord2", "mainaccord3"]].isin(
    dislike_accords
).any(axis=1)

perfume_df = perfume_df[~mask_exclude].reset_index(drop=True)

# =========================================================
# 3. 스타일 예측
# =========================================================
df_1 = user_df.drop(columns=["계절", "비선호_향조"], axis=1)


def merge_clothes(base_df, clothes_df, clothes_type):
    cols = [
        "식별자",
        f"{clothes_type}_소재",
        f"{clothes_type}_핏",
        f"{clothes_type}_프린트",
        f"{clothes_type}_디테일",
        f"{clothes_type}_넥라인",
        "서브스타일",
    ]

    if clothes_type == "상의":
        cols.append("상의_소매기장")
    else:
        cols.append(f"{clothes_type}_기장")

    cols = [c for c in cols if c in clothes_df.columns]

    merged = base_df.merge(
        clothes_df[cols],
        left_on=f"{clothes_type}_식별자",
        right_on="식별자",
        how="left",
    ).drop(columns=["식별자"], errors="ignore")

    return merged.rename(columns={"서브스타일": f"{clothes_type}_서브스타일"})


df_1 = merge_clothes(df_1, top_bottom, "상의")
df_1 = merge_clothes(df_1, top_bottom, "하의")
df_1 = merge_clothes(df_1, onepiece, "원피스")

df_1["상의_카테고리"] = df_1["상의_카테고리"].replace({"브라탑": "탑"})

BASE_PATH = "code/2_data analysis/clothes/"

model_0 = joblib.load(BASE_PATH + "0_style_model.pkl")
encoder_0 = joblib.load(BASE_PATH + "0_clothes_encoder.pkl")
label_encoder_0 = joblib.load(BASE_PATH + "0_style_label_encoder.pkl")

model_1 = joblib.load(BASE_PATH + "1_style_model.pkl")
encoder_1 = joblib.load(BASE_PATH + "1_clothes_encoder.pkl")
label_encoder_1 = joblib.load(BASE_PATH + "1_style_label_encoder.pkl")

row = df_1.iloc[-1]

if pd.isna(row["원피스_식별자"]):
    model, encoder, label_encoder = model_0, encoder_0, label_encoder_0
    row["색상_조합"] = f"{row['상의_색상']}_{row['하의_색상']}"
    row["핏_조합"] = f"{row['상의_핏']}_{row['하의_핏']}"
else:
    model, encoder, label_encoder = model_1, encoder_1, label_encoder_1

train_cols = encoder.feature_names_in_

row_df = pd.DataFrame([row[train_cols]])
row_df[train_cols] = encoder.transform(row_df[train_cols].astype("object"))

user_style = label_encoder.inverse_transform([model.predict(row_df)[0]])[0]
user_df.loc[user_df["사용자_식별자"] == user_id, "예측_스타일"] = user_style

# =========================================================
# 4. 스타일 점수 (향조 필터링)
# =========================================================
style_fragrance_score = {
    "로맨틱": {
        "플로럴향, 달콤한향": 46.15,
        "싱그러운 풀 향": 7.69,
        "머스크같은 중후한향": 0.0,
        "파우더느낌의 부드러운향": 30.77,
        "시원하고 신선한 바다 향": 15.38,
        "감귤류의 상큼한 향": 0.0,
        "라벤더같은 상쾌한향": 0.0,
    },
    "섹시": {
        "플로럴향, 달콤한향": 20.0,
        "싱그러운 풀 향": 40.0,
        "머스크같은 중후한향": 40.0,
        "파우더느낌의 부드러운향": 0.0,
        "시원하고 신선한 바다 향": 0.0,
        "감귤류의 상큼한 향": 0.0,
        "라벤더같은 상쾌한향": 0.0,
    },
    "소피스트케이티드": {
        "플로럴향, 달콤한향": 30.0,
        "싱그러운 풀 향": 10.0,
        "머스크같은 중후한향": 10.0,
        "파우더느낌의 부드러운향": 40.0,
        "시원하고 신선한 바다 향": 10.0,
        "감귤류의 상큼한 향": 0.0,
        "라벤더같은 상쾌한향": 0.0,
    },
    "스포티": {
        "플로럴향, 달콤한향": 14.29,
        "싱그러운 풀 향": 9.52,
        "머스크같은 중후한향": 0.0,
        "파우더느낌의 부드러운향": 4.76,
        "시원하고 신선한 바다 향": 57.14,
        "감귤류의 상큼한 향": 14.29,
        "라벤더같은 상쾌한향": 0.0,
    },
    "클래식": {
        "플로럴향, 달콤한향": 9.09,
        "싱그러운 풀 향": 12.12,
        "머스크같은 중후한향": 6.06,
        "파우더느낌의 부드러운향": 21.21,
        "시원하고 신선한 바다 향": 36.36,
        "감귤류의 상큼한 향": 6.06,
        "라벤더같은 상쾌한향": 9.09,
    },
    "젠더리스": {
        "플로럴향, 달콤한향": 21.43,
        "싱그러운 풀 향": 21.43,
        "머스크같은 중후한향": 0.0,
        "파우더느낌의 부드러운향": 28.57,
        "시원하고 신선한 바다 향": 14.29,
        "감귤류의 상큼한 향": 14.29,
        "라벤더같은 상쾌한향": 0.0,
    },
    "아방가르드": {
        "플로럴향, 달콤한향": 11.11,
        "싱그러운 풀 향": 5.56,
        "머스크같은 중후한향": 0.0,
        "파우더느낌의 부드러운향": 16.67,
        "시원하고 신선한 바다 향": 44.44,
        "감귤류의 상큼한 향": 16.67,
        "라벤더같은 상쾌한향": 5.56,
    },
}
d = style_fragrance_score[user_style]

value_to_keys = defaultdict(list)
for k, v in d.items():
    value_to_keys[v].append(k)

top_keys = [
    k
    for v in sorted(value_to_keys.keys(), reverse=True)[:2]
    for k in value_to_keys[v]
]

score_df = (
    perfume_classification
    .loc[perfume_classification["fragrance"].isin(top_keys), ["perfume_id"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

# =========================================================
# B. 색상 점수
# =========================================================
def parse_rgb(x):
    nums = re.findall(r"\d+", str(x))
    return tuple(map(int, nums[:3]))


clothes_color["rgb"] = clothes_color["rgb_tuple"].apply(parse_rgb)
perfume_color["rgb"] = perfume_color["color"].apply(parse_rgb)

clothes_color_map = dict(zip(clothes_color["color"], clothes_color["rgb"]))
perfume_color_map = dict(zip(perfume_color["mainaccord"], perfume_color["rgb"]))


def calc_color_score(c_vec, f_vec):
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(c_vec, f_vec)))
    return 100 * (1 - dist / (255 * math.sqrt(3)))


def mix_fragrance(a1, a2, a3):
    return [
        a1[i] * 0.6 + a2[i] * 0.3 + a3[i] * 0.1
        for i in range(3)
    ]


if pd.notna(user_row["원피스_색상"]):
    clothes_vec = clothes_color_map[user_row["원피스_색상"]]
else:
    top = clothes_color_map[user_row["상의_색상"]]
    bottom = clothes_color_map[user_row["하의_색상"]]
    clothes_vec = [top[i] * 0.7 + bottom[i] * 0.3 for i in range(3)]

perfume_df["color_score"] = perfume_df.apply(
    lambda r: calc_color_score(
        clothes_vec,
        mix_fragrance(
            perfume_color_map[r["mainaccord1"]],
            perfume_color_map[r["mainaccord2"]],
            perfume_color_map[r["mainaccord3"]],
        ),
    ),
    axis=1,
)

score_df = score_df.merge(
    perfume_df[["perfume_id", "color_score"]],
    on="perfume_id",
    how="inner"
)
# =========================================================
# C. 계절 점수
# =========================================================
season_map = {"봄": "spring", "여름": "summer", "가을": "fall", "겨울": "winter"}
user_season = season_map[user_row["계절"]]

perfume_season["season_score"] = (
    perfume_season[user_season]
    / perfume_season[["spring", "summer", "fall", "winter"]].sum(axis=1)
).fillna(0) * 100

score_df = score_df.merge(
    perfume_season[["perfume_id", "season_score"]],
    on="perfume_id",
    how="inner"
)

# =========================================================
# 7. 최종 점수
# =========================================================
score_df["myscore"] = (
    score_df["color_score"] + score_df["season_score"]
)

score_df = score_df.sort_values("myscore", ascending=False).reset_index(drop=True)
score_df["user_style"] = user_style

score_df
# score_df.to_csv("data/03_results/recommendation/score.csv",
#     index=False,encoding="utf-8-sig"
# )