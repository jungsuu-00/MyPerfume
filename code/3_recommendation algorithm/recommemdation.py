import pandas as pd
import pandas as pd
import math
import re
import joblib


def recommend_perfumes(
    user_info: list[dict],
    perfume_classification: list[dict],
    perfume: list[dict],
    perfume_season: list[dict],
    상의_하의: list[dict],
    원피스: list[dict],
    clothes_color: list[dict],
    perfume_color: list[dict],
) -> list[dict]:

    user_df = pd.DataFrame(user_info)
    df_perfume = pd.DataFrame(perfume_classification)
    perfume_df = pd.DataFrame(perfume)
    season_df = pd.DataFrame(perfume_season)
    df_clothes = pd.DataFrame(상의_하의)
    df_onepiece = pd.DataFrame(원피스)
    clothes_color_df = pd.DataFrame(clothes_color)
    perfume_color_df = pd.DataFrame(perfume_color)

    # =========================================================
    # 최신 사용자 선택
    # =========================================================
    user_row = user_df.iloc[-1]
    user_id = user_row["사용자_식별자"]

    # =========================================================
    # 비선호 향조 제외
    # =========================================================
    # 사용자의 비선호_향조 추출
    dislike_raw = user_df.loc[user_df["사용자_식별자"] == user_id, "비선호_향조"].iloc[
        0
    ]

    # 리스트로 변경
    if isinstance(dislike_raw, str):
        dislike_accords = [x.strip() for x in dislike_raw.split(",")]
    elif isinstance(dislike_raw, list):
        dislike_accords = dislike_raw
    else:
        dislike_accords = []

    # perfume에서 제외 조건 만들기
    mask_exclude = (
        perfume_df[["mainaccord1", "mainaccord2", "mainaccord3"]]
        .isin(dislike_accords)
        .any(axis=1)
    )

    # 비선호 향조가 포함된 향수 제외
    perfume_df = perfume_df[~mask_exclude].reset_index(drop=True)
    # =========================================================
    # A. 스타일 & 향조
    # =========================================================

    df_1 = user_df.drop(columns=["계절", "비선호_향조"], axis=1)

    def merge_clothes(user_df, clothes_df, clothes_type):
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

        user_df = user_df.merge(
            clothes_df[cols],
            left_on=f"{clothes_type}_식별자",
            right_on="식별자",
            how="left",
        ).drop(columns=["식별자"], errors="ignore")

        user_df = user_df.rename(columns={"서브스타일": f"{clothes_type}_서브스타일"})
        return user_df

    df_1 = merge_clothes(df_1, df_clothes, "상의")
    df_1 = merge_clothes(df_1, df_clothes, "하의")
    df_1 = merge_clothes(df_1, df_onepiece, "원피스")

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
        train_cols = encoder.feature_names_in_
    else:
        model, encoder, label_encoder = model_1, encoder_1, label_encoder_1
        train_cols = encoder.feature_names_in_

    row_df = pd.DataFrame([row[train_cols]])
    row_df[train_cols] = encoder.transform(row_df[train_cols].astype("object"))

    user_style = label_encoder.inverse_transform([model.predict(row_df)[0]])[0]

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

    df_perfume["style_score"] = df_perfume["fragrance"].apply(
        lambda x: style_fragrance_score.get(user_style, {}).get(x, 0)
    )

    # =========================================================
    # B. 색상
    # =========================================================
    def parse_rgb(x):
        nums = re.findall(r"\d+", str(x))
        return tuple(map(int, nums[:3]))

    clothes_color_df["rgb"] = clothes_color_df["rgb_tuple"].apply(parse_rgb)
    perfume_color_df["rgb"] = perfume_color_df["color"].apply(parse_rgb)

    clothes_color = dict(zip(clothes_color_df["color"], clothes_color_df["rgb"]))
    perfume_color = dict(zip(perfume_color_df["mainaccord"], perfume_color_df["rgb"]))

    def calc_color_score(c_vec, f_vec):
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(c_vec, f_vec)))
        return 100 * (1 - dist / (255 * math.sqrt(3)))

    def mix_fragrance(a1, a2, a3):
        return [a1[i] * 0.6 + a2[i] * 0.3 + a3[i] * 0.1 for i in range(3)]

    if pd.notna(user_row["원피스_색상"]):
        clothes_vec = clothes_color[user_row["원피스_색상"]]
    else:
        top = clothes_color[user_row["상의_색상"]]
        bottom = clothes_color[user_row["하의_색상"]]
        clothes_vec = [top[i] * 0.7 + bottom[i] * 0.3 for i in range(3)]

    perfume_df["color_score"] = perfume_df.apply(
        lambda r: calc_color_score(
            clothes_vec,
            mix_fragrance(
                perfume_color[r["mainaccord1"]],
                perfume_color[r["mainaccord2"]],
                perfume_color[r["mainaccord3"]],
            ),
        ),
        axis=1,
    )

    # =========================================================
    # C. 계절
    # =========================================================
    # season_map = {"봄": "spring", "여름": "summer", "가을": "fall", "겨울": "winter"}
    user_season = season_map[user_row["계절"]]

    season_df["season_score"] = season_df[user_season] / (
        season_df[["spring", "summer", "fall", "winter"]].sum(axis=1)
    )
    season_df["season_score"] = season_df["season_score"].fillna(0) * 100

    perfume_df = perfume_df.merge(
        season_df[["perfume_id", "season_score"]], on="perfume_id", how="left"
    )
    # =========================================================
    # 7. 최종 점수
    # =========================================================
    final_df = df_perfume.merge(
        perfume_df[["perfume_id", "color_score", "season_score"]], on="perfume_id"
    )

    final_df["myscore"] = (
        final_df["style_score"] + final_df["color_score"] + final_df["season_score"]
    )

    final_result = final_df.sort_values("myscore", ascending=False).reset_index(
        drop=True
    )
    final_result.drop(columns=["fragrance"], inplace=True)
    final_result = final_result.head(3)

    return final_result.to_dict(orient="records")
