from ui.models import (
    UserSmellingInput,UserSmellingMyScore,
    UserInfo, Perfume, PerfumeClassification,
    PerfumeColor, PerfumeSeason, Score,
    TopBottom, Dress, ClothesColor )
from django.db.models import Q
import numpy as np
import pandas as pd
import math, re
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from django.conf import settings
import warnings
warnings.filterwarnings("ignore")
import itertools

# =========================================================
# 모델 로딩: 프로젝트 루트 경로 내 ml_models 폴더에서 학습된 모델을 로드합니다.
# =========================================================
BASE_PATH = os.path.join(settings.BASE_DIR, 'ui', 'recommend', 'models')

# 0: 상하의(투피스)용 모델, 1: 원피스용 모델
model_0 = joblib.load(os.path.join(BASE_PATH, "0_style_model.pkl"))
encoder_0 = joblib.load(os.path.join(BASE_PATH, "0_clothes_encoder.pkl"))
label_encoder_0 = joblib.load(os.path.join(BASE_PATH, "0_style_label_encoder.pkl"))

model_1 = joblib.load(os.path.join(BASE_PATH, "1_style_model.pkl"))
encoder_1 = joblib.load(os.path.join(BASE_PATH, "1_clothes_encoder.pkl"))
label_encoder_1 = joblib.load(os.path.join(BASE_PATH, "1_style_label_encoder.pkl"))


# =========================================================
# [기능] 색상 문자열 파싱
# 설명: DB의 '#FFFFFF' 또는 'rgb(255,255,255)' 문자열을 숫자 튜플 (R, G, B)로 변환합니다.
# =========================================================
def parse_rgb(x):
    if not x:
        raise ValueError("❌ [데이터 누락] DB에 색상 값이 비어 있는 행이 있습니다.")

    x_str = str(x).strip()

    # 1. 헥사코드 처리 (#CCCCCC 등)
    if x_str.startswith('#'):
        hex_val = x_str.lstrip('#')
        if len(hex_val) == 6:
            return tuple(int(hex_val[i:i+2], 16) for i in (0, 2, 4))
        else:
            raise ValueError(f"❌ [데이터 오류] 잘못된 헥사코드 형식: '{x_str}'")

    # 2. 숫자 기반 형식 처리 (rgb(...) 또는 (r,g,b))
    nums = list(map(int, re.findall(r"\d+", x_str)))
    if len(nums) >= 3:
        return tuple(nums[:3])

    raise ValueError(f"❌ [데이터 오류] 지원하지 않는 색상 형식입니다: '{x_str}'.")


# =========================================================
# [기능] 향수 색상 혼합
# 설명: 향수의 상위 3개 어코드 색상을 6:3:1 비율로 혼합하여 대표 RGB 벡터를 생성
# =========================================================
def mix_rgb(a1, a2, a3):
    # 인덱스 에러 방지를 위한 Strict 체크
    for idx, color in enumerate([a1, a2, a3], 1):
        if not (isinstance(color, (tuple, list)) and len(color) >= 3):
            raise ValueError(f"❌ [데이터 오류] {idx}번째 향조의 색상 데이터가 손상되었습니다.")

    return [a1[i] * 0.6 + a2[i] * 0.3 + a3[i] * 0.1 for i in range(3)]


# =========================================================
# [기능] 색상 점수 계산
# 설명: 사용자의 옷 색상 벡터와 향수의 색상 벡터 간의 유클리드 거리를 측정하여 100점 만점으로 환산
# =========================================================
def calc_color_score(c_vec, f_vec):
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(c_vec, f_vec)))
    return 100 * (1 - dist / (255 * math.sqrt(3)))

# =====================================================
def find_best_weights():
    k = 3
    # ✅ float 변환 추가
    final_result = pd.DataFrame.from_records(
        UserSmellingMyScore.objects.all().values()
    )
    # Decimal 컬럼들을 float로 변환
    numeric_cols = ['style_score', 'color_score', 'season_score']
    for col in numeric_cols:
        if col in final_result.columns:
            final_result[col] = final_result[col].astype(float)
    
    user_smelling_df = pd.DataFrame.from_records(
        UserSmellingInput.objects.all().values()
    )
    
    # ✅ 디버깅: 실제 컬럼명 확인
    print("UserSmellingInput 컬럼:", user_smelling_df.columns.tolist())
    print("UserSmellingMyScore 컬럼:", final_result.columns.tolist())

    weight_candidates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    weight_results = []

    # =========================================================
    # 1. 가중치 조합 탐색
    # =========================================================
    for w_style, w_color, w_season in itertools.product(weight_candidates, repeat=3):

        # 전부 0이면 스킵
        if w_style + w_color + w_season == 0:
            continue

        # 합이 1이 되도록 정규화
        total = w_style + w_color + w_season
        w_style /= total
        w_color /= total
        w_season /= total

        df = final_result.copy()
        df["myscore"] = (
            df["style_score"] * w_style +
            df["color_score"] * w_color +
            df["season_score"] * w_season
        )

        # =========================================================
        # 2. Precision@k 계산
        # =========================================================
        topk_df = (
            df.groupby("user_id", group_keys=False)
              .apply(lambda x: x.nlargest(k, "myscore"))
        )

        precisions = []

        for uid in topk_df["user_id"].unique():
            recs = topk_df[topk_df["user_id"] == uid]["perfume_id"].tolist()
            
            # ✅ 실제 컬럼명으로 수정 (perfume_id 또는 perfume_id_id 등)
            # 만약 컬럼명이 'perfume_id_id'라면 아래처럼 수정
            perfume_col = 'perfume_id' if 'perfume_id' in user_smelling_df.columns else 'perfume_id_id'
            user_col = 'smelling_user_id' if 'smelling_user_id' in user_smelling_df.columns else 'user_id'
            
            actuals = user_smelling_df[
                user_smelling_df[user_col] == uid
            ][perfume_col].tolist()

            if len(actuals) == 0:
                continue

            hit_count = len(set(recs) & set(actuals))
            precisions.append(hit_count / k)

        if len(precisions) == 0:
            mean_prec = 0
        else:
            mean_prec = sum(precisions) / len(precisions)

        weight_results.append({
            "w_style": w_style,
            "w_color": w_color,
            "w_season": w_season,
            "mean_precision": mean_prec
        })

    # =========================================================
    # 3. 결과 정리
    # =========================================================
    weight_df = pd.DataFrame(weight_results)

    best_row = weight_df.loc[
        weight_df["mean_precision"].idxmax()
    ]

    best_weights = {
        "w_style": best_row["w_style"],
        "w_color": best_row["w_color"],
        "w_season": best_row["w_season"],
        "mean_precision": best_row["mean_precision"]
    }

    return best_weights


def myscore_cal(user_id: int) -> list[Score]:
    print(f"\n{'=' * 60}")
    print(f"🚀 myscore_cal 시작: user_id={user_id}")
    print(f"{'=' * 60}\n")

    # ---------------------------------------------------------
    # 0. 사용자 조회: UserInfo 테이블에서 사용자 정보 설정
    # ---------------------------------------------------------
    user_row = UserInfo.objects.get(user_id=user_id)
    print(f"✅ 사용자 조회 성공: {user_row}")

    dislike_accords = (
        [x.strip() for x in user_row.disliked_accord.split(",")]
        if user_row.disliked_accord else []
    )

    # ---------------------------------------------------------
    # 1. 향수 필터링: 사용자가 설정한 비선호 향조를 포함하는 향수를 1차적으로 제외
    # ---------------------------------------------------------
    print("\nSTEP 1: 향수 필터링 (비선호 향조 제외)")
    perfume_qs = Perfume.objects.exclude(
        Q(mainaccord1__in=dislike_accords) |
        Q(mainaccord2__in=dislike_accords) |
        Q(mainaccord3__in=dislike_accords)
    )
    perfume_df = pd.DataFrame.from_records(perfume_qs.values())

    if perfume_df.empty:
        raise ValueError("❌ 필터링 후 조건에 맞는 향수가 없습니다.")

    # ---------------------------------------------------------
    # 2. 사용자 의류 정보 병합: 선택한 상하의 또는 원피스 데이터를 모델 예측용 데이터프레임 만듬
    # ---------------------------------------------------------
    print("\nSTEP 2: 사용자 의류 정보 병합")
    df_row = pd.DataFrame([{}])

    def merge_clothes(df, model_cls, obj_id, prefix):
        """DB 필드에서 직접 데이터를 추출하여 셋팅 (기본값 없음)"""
        print(f"🔍 {prefix} 병합 중 (ID: {obj_id})...")
        clothes = model_cls.objects.get(pk=obj_id)

        if prefix == "상의":
            df["상의_카테고리"] = clothes.top_category
            df["상의_색상"] = clothes.top_color.color  # 색상 데이터 필수
            df["상의_소매기장"] = clothes.top_sleeve_length
            df["상의_소재"] = clothes.top_material
            df["상의_프린트"] = clothes.top_print
            df["상의_넥라인"] = clothes.top_neckline
            df["상의_핏"] = clothes.top_fit
            df["상의_서브스타일"] = clothes.sub_style
        elif prefix == "하의":
            df["하의_카테고리"] = clothes.bottom_category
            df["하의_색상"] = clothes.bottom_color.color
            df["하의_기장"] = clothes.bottom_length
            df["하의_소재"] = clothes.bottom_material
            df["하의_핏"] = clothes.bottom_fit
            df["하의_서브스타일"] = clothes.sub_style
        elif prefix == "원피스":
            df["원피스_기장"] = clothes.dress_length
            df["원피스_색상"] = clothes.dress_color.color
            df["원피스_소매기장"] = clothes.dress_sleeve_length
            df["원피스_소재"] = clothes.dress_material
            df["원피스_프린트"] = clothes.dress_print
            df["원피스_핏"] = clothes.dress_fit
            df["원피스_넥라인"] = clothes.dress_neckline
            df["원피스_디테일"] = clothes.dress_detail
            df["원피스_서브스타일"] = clothes.sub_style
        return df

    if user_row.top_id_id:
        df_row = merge_clothes(df_row, TopBottom, user_row.top_id_id, "상의")
    if user_row.bottom_id_id:
        df_row = merge_clothes(df_row, TopBottom, user_row.bottom_id_id, "하의")
    if user_row.dress_id_id:
        df_row = merge_clothes(df_row, Dress, user_row.dress_id_id, "원피스")

    if "상의_카테고리" in df_row.columns:
        df_row["상의_카테고리"] = df_row["상의_카테고리"].replace({"브라탑": "탑"})

    # ---------------------------------------------------------
    # 3. 스타일 예측: 학습된 머신러닝 모델을 사용하여 현재 코디의 스타일을 예측
    # ---------------------------------------------------------
    print("\nSTEP 3: 스타일 예측")
    if not user_row.dress_id_id:
        model, encoder, label_encoder = model_0, encoder_0, label_encoder_0
        df_row["색상_조합"] = df_row["상의_색상"].astype(str) + "_" + df_row["하의_색상"].astype(str)
        df_row["핏_조합"] = df_row["상의_핏"].astype(str) + "_" + df_row["하의_핏"].astype(str)
    else:
        model, encoder, label_encoder = model_1, encoder_1, label_encoder_1

    # 인코더를 통해 변환 후 다시 DataFrame으로 만들어 컬럼 이름표를 유지 (UserWarning 방지)
    raw_encoded = encoder.transform(df_row[list(encoder.feature_names_in_)].astype("object"))
    encoded_df = pd.DataFrame(raw_encoded, columns=encoder.get_feature_names_out())

    # 모델 예측 실행
    user_style = label_encoder.inverse_transform([model.predict(encoded_df)[0]])[0]
    print(f"✅ 예측된 스타일: {user_style}")

    # ---------------------------------------------------------
    # 4. 스타일 기반 향수 필터링: 예측된 스타일에 어울리는 향조(Accords) 점수를 매핑하고 해당 향수만 추출
    # ---------------------------------------------------------
    print("\nSTEP 4: 스타일 기반 향수 필터링")
    style_fragrance_score = {
        "로맨틱": {
            "플로럴향, 달콤한향": 7,
            "싱그러운 풀 향": 4,
            "머스크같은 중후한향": 2,
            "파우더느낌의 부드러운향": 6,
            "시원하고 신선한 바다 향": 5,
            "감귤류의 상큼한 향": 2,
            "라벤더같은 상쾌한향": 2,
        },
        "섹시": {
            "플로럴향, 달콤한향": 5,
            "싱그러운 풀 향": 6.5,
            "머스크같은 중후한향": 6.5,
            "파우더느낌의 부드러운향": 3,
            "시원하고 신선한 바다 향": 3,
            "감귤류의 상큼한 향": 3,
            "라벤더같은 상쾌한향": 3,
        },
        "소피스트케이티드": {
            "플로럴향, 달콤한향": 6,
            "싱그러운 풀 향": 4,
            "머스크같은 중후한향": 4,
            "파우더느낌의 부드러운향": 7,
            "시원하고 신선한 바다 향": 4,
            "감귤류의 상큼한 향": 1.5,
            "라벤더같은 상쾌한향": 1.5,
        },
        "스포티": {
            "플로럴향, 달콤한향": 5,
            "싱그러운 풀 향": 4,
            "머스크같은 중후한향": 2,
            "파우더느낌의 부드러운향": 3,
            "시원하고 신선한 바다 향": 7,
            "감귤류의 상큼한 향": 5,
            "라벤더같은 상쾌한향": 2,
        },
        "클래식": {
            "플로럴향, 달콤한향": 3.5,
            "싱그러운 풀 향": 4.5,
            "머스크같은 중후한향": 2,
            "파우더느낌의 부드러운향": 6,
            "시원하고 신선한 바다 향": 7,
            "감귤류의 상큼한 향": 2,
            "라벤더같은 상쾌한향": 3.5,
        },
        "젠더리스": {
            "플로럴향, 달콤한향": 5.5,
            "싱그러운 풀 향": 5.5,
            "머스크같은 중후한향": 2,
            "파우더느낌의 부드러운향": 7,
            "시원하고 신선한 바다 향": 4,
            "감귤류의 상큼한 향": 4,
            "라벤더같은 상쾌한향": 2,
        },
        "아방가르드": {
            "플로럴향, 달콤한향": 4,
            "싱그러운 풀 향": 2.5,
            "머스크같은 중후한향": 1,
            "파우더느낌의 부드러운향": 5.5,
            "시원하고 신선한 바다 향": 7,
            "감귤류의 상큼한 향": 5.5,
            "라벤더같은 상쾌한향": 2.5,
        }
    }

    style_scores = style_fragrance_score[user_style]  # 매핑 실패 시 즉시 에러

    top_fragrances = list(style_scores.keys())
    classification_df = pd.DataFrame.from_records(PerfumeClassification.objects.all().values())

    filtered_ids = classification_df[classification_df["fragrance"].isin(top_fragrances)]["perfume_id"].unique()
    perfume_df = perfume_df[perfume_df["perfume_id"].isin(filtered_ids)]
    print(f"✅ 스타일 필터링 후 향수 개수: {len(perfume_df)}")

    # ---------------------------------------------------------
    # 5. 색상 정보 준비: 옷과 향수의 대표 색상을 RGB 벡터화
    # ---------------------------------------------------------
    print("\nSTEP 5: 색상 점수 준비")
    clothes_color_map = {c.color: parse_rgb(c.rgb_tuple) for c in ClothesColor.objects.all()}
    perfume_color_map = {c.mainaccord: parse_rgb(c.color) for c in PerfumeColor.objects.all()}

    # 사용자의 옷 색상 벡터 계산 (투피스 시 7:3 가중치 혼합)
    if user_row.dress_id_id:
        clothes_vec = clothes_color_map[df_row["원피스_색상"].iloc[0]]
    else:
        top_rgb = clothes_color_map[df_row["상의_색상"].iloc[0]]
        bottom_rgb = clothes_color_map[df_row["하의_색상"].iloc[0]]
        clothes_vec = [top_rgb[i] * 0.7 + bottom_rgb[i] * 0.3 for i in range(3)]

    # ---------------------------------------------------------
    # 6. 계절 점수 계산: 사용자의 계절 설정(한글/영어 모두 대응)을 기반으로 계절 조화 점수를 미리 계산합니다.
    # ---------------------------------------------------------
    print("\nSTEP 6: 계절 점수 계산")
    season_df = pd.DataFrame.from_records(PerfumeSeason.objects.all().values())
    season_map = {
        "봄": "spring", "여름": "summer", "가을": "fall", "겨울": "winter",
        "spring": "spring", "summer": "summer", "fall": "fall", "winter": "winter"
    }
    user_season = season_map[user_row.season]

    # ---------------------------------------------------------
    # 7. 최종 점수 합산: 개별 향수마다 스타일, 색상, 계절 점수를 합산하여 Score 객체를 생성합니다.
    # ---------------------------------------------------------
    print("\nSTEP 7: 최종 점수 계산 및 Score 객체 생성")
    score_list = []
    fragrance_dict = dict(zip(classification_df['perfume_id'], classification_df['fragrance']))

    # =========================
    # 1차 패스: 원점수 수집
    # =========================
    style_raw = []
    color_raw = []
    season_raw = []
    perfume_ids = []

    for idx, (_, p_row) in enumerate(perfume_df.iterrows(), 1):
        p_id = p_row["perfume_id"]
        perfume_ids.append(p_id)

        # 7-1. 스타일 점수
        p_fragrance = fragrance_dict[p_id]
        calc_style_score = style_scores[p_fragrance]
        style_raw.append(calc_style_score)

        # 7-2. 색상 점수
        # Pandas DataFrame에서는 FK 필드명이 'mainaccord1_id' 형식이 됨을 유의
        a1 = p_row["mainaccord1_id"]
        a2 = p_row["mainaccord2_id"]
        a3 = p_row["mainaccord3_id"]

        color_vec = mix_rgb(perfume_color_map[a1], perfume_color_map[a2], perfume_color_map[a3])
        color_score = calc_color_score(clothes_vec, color_vec)
        color_raw.append(color_score)

        # 7-3. 계절 점수
        s_row = season_df[season_df["perfume_id"] == p_id].iloc[0]
        total_season_val = s_row[["spring", "summer", "fall", "winter"]].sum()
        season_score = (s_row[user_season] / total_season_val * 100) if total_season_val > 0 else 0
        season_raw.append(season_score)

    # numpy 변환
    style_raw = np.array(style_raw).reshape(-1, 1)
    color_raw = np.array(color_raw).reshape(-1, 1)
    season_raw = np.array(season_raw).reshape(-1, 1)

    # =========================
    #  MinMaxScaler 적용
    # =========================
    style_mm = MinMaxScaler().fit_transform(style_raw)
    color_mm = MinMaxScaler().fit_transform(color_raw)
    season_mm = MinMaxScaler().fit_transform(season_raw)

    best_weights = find_best_weights()

    w_style = best_weights["w_style"]
    w_color = best_weights["w_color"]
    w_season = best_weights["w_season"]

    # =========================
    # 2차 패스: myscore 계산 & 저장 (ε smoothing 적용)
    # =========================
    EPS = 0.02  # ε smoothing 값
    for idx, p_id in enumerate(perfume_ids, 1):
        s = (float(style_mm[idx - 1][0]) + EPS) / (1 + EPS)
        c = (float(color_mm[idx - 1][0]) + EPS) / (1 + EPS)
        se = (float(season_mm[idx - 1][0]) + EPS) / (1 + EPS)
        myscore = w_style*s + w_color*c + w_season*se

        if idx <= 3:
            print(
                f"향수 #{idx} (ID:{p_id}): "
                f"Style({s:.3f}) + Color({c:.3f}) + Season({se:.3f}) = {myscore:.3f}"
            )

        score_list.append(
            Score(
                user=user_row,
                perfume_id=p_id,
                style_score=s,
                color_score=c,
                season_score=se,
                myscore=myscore,
                user_style=user_style
            )
        )

    # ---------------------------------------------------------
    # 8. 리턴: myscore 기준 내림차순 정렬 후 상위 3개 객체 리스트를 반환
    # ---------------------------------------------------------
    top3 = sorted(score_list, key=lambda x: x.myscore, reverse=True)[:3]

    print(f"\n{'=' * 60}")
    print("🏆 Top3 결과")
    print(f"{'=' * 60}")
    for i, score in enumerate(top3, 1):
        print(f"{i}. Perfume ID: {score.perfume_id}, myscore: {score.myscore:.2f}")

    print(f"\n✅ myscore_cal 완료\n")

    return top3
