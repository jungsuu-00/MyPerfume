import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 데이터 로드
score_df = pd.read_csv("data/03_results/recommendation/score.csv")
user_df = pd.read_csv("data/03_results/clothes/user_info.csv")
perfume_df = pd.read_csv("data/03_results/perfume/perfume.csv")
classification_df = pd.read_csv("data/03_results/perfume/perfume_classification.csv")
color_df = pd.read_csv("data/03_results/perfume/perfume_color.csv")
season_df = pd.read_csv("data/03_results/perfume/perfume_season.csv")

# 추천 향수 3개의 LLM 입력 데이터 생성
## A) 사용자 
def build_user_context(user_df: pd.DataFrame):
    user = user_df.iloc[-1]

    user_style_text = []
    if pd.notna(user["상의_색상"]):
        user_style_text.append(f"상의는 {user['상의_색상']} 계열")
    if pd.notna(user["하의_색상"]):
        user_style_text.append(f"하의는 {user['하의_색상']} 계열")
    if pd.notna(user["원피스_색상"]):
        user_style_text.append(f"원피스는 {user['원피스_색상']} 계열")

    user_style_summary = ", ".join(user_style_text)

    return {
        "user_season": user["계절"],
        "user_style": f"전체적으로 {user_style_summary}의 차분하고 부드러운 스타일",
        "disliked_accords": user["비선호_향조"]
    }
    
## B) 향수 
def build_llm_input_for_perfume(
    score_row,
    perfume_df,
    classification_df,
    color_df,
    season_df,
    user_context
):
    perfume_id = score_row["perfume_id"]

    perfume = perfume_df[perfume_df["perfume_id"] == perfume_id].iloc[0]
    accords = classification_df[classification_df["perfume_id"] == perfume_id].iloc[0]
    season_info = season_df[season_df["perfume_id"] == perfume_id].iloc[0]

    # ✅ 1) 향 결 설명 (사람 언어)
    fragrance_desc = accords["fragrance"]

    # ✅ 2) 계절 적합도 상위 2개 추출
    season_scores = {
        "봄": season_info["spring"],
        "여름": season_info["summer"],
        "가을": season_info["fall"],
        "겨울": season_info["winter"],
    }

    top_seasons = sorted(
        season_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:2]

    season_desc = ", ".join([s[0] for s in top_seasons])

    return {
        "perfume_name": perfume["Perfume"],
        "brand": perfume["Brand"],

        "my_score": score_row["myscore"],
        "style_score": score_row["style_score"],
        "color_score": score_row["color_score"],
        "season_score": score_row["season_score"],

        "user_style": score_row["user_style"],
        "user_season": user_context["user_season"],

        # 🔥 핵심 추가
        "fragrance_desc": fragrance_desc,          # 예: 플로럴향, 달콤한향
        "best_seasons": season_desc,               # 예: 가을, 겨울

        "perfume_mainaccords": ", ".join([
            perfume["mainaccord1"],
            perfume["mainaccord2"],
            perfume["mainaccord3"]
        ]),

        "review_summary": "(리뷰없음)"
    }
## A+B) 종합
def build_top3_llm_inputs(
    score_df,
    user_df,
    perfume_df,
    classification_df,
    color_df,
    season_df
):
    user_context = build_user_context(user_df)

    llm_inputs = []
    for _, row in score_df.iterrows():
        llm_input = build_llm_input_for_perfume(
            row,
            perfume_df,
            classification_df,
            color_df,
            season_df,
            user_context
        )
        llm_inputs.append(llm_input)

    return llm_inputs

# LLM 호출하여 종합 추천 이유 생성
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_top3_recommend_summary(
    user_style: str,
    user_season: str,
    perfumes: list
):
    system_prompt = """
    너는 향수 추천 서비스에서 종합 추천 이유를 작성하는 에디터다.

    다음 구조를 반드시 지켜서 작성한다.

    1. 총 3개의 문단으로만 작성한다.
    2. 각 문단은 한 줄 이상 띄우지 않는다.
    3. 각 문단은 명확한 역할을 가진다.

    - 1문단:
    왜 이 세 가지 향수가 함께 추천되었는지에 대한 전체 요약.
    사용자의 스타일과 계절을 중심으로 공통된 방향성을 설명한다.

    - 2문단:
    style / color / season 관점에서 공통적으로 작용한 요소를 구체적으로 풀어 설명한다.
    이때 color는 반드시 아래 순서로 자세히 설명한다.

    (필수 색감 서술 규칙)
    A) 사용자 착장의 색감(상의/하의/원피스)을 먼저 구체적으로 묘사한다.
    - 밝기(밝은/중간/짙은), 채도(선명한/차분한), 온도감(웜/쿨), 대비(톤온톤/대비감) 중 최소 2가지를 포함한다.
    B) 그 색감이 향수의 분위기(따뜻함/차분함/세련됨/생동감 등)와 어떻게 이어지는지 설명한다.
    C) 문장만 나열하지 말고, “어떤 장면에서 어울리는지”를 짧게 한 번 붙인다. (예: 가을 오후 산책, 운동 후 데일리 등)

    점수의 수치나 비교 표현은 사용하지 않는다.
    같은 의미의 문장을 반복하지 않는다.

    - 3문단:
    세 향수가 서로 어떤 결의 차이를 가지는지 한 문장씩 간결하게 정리한다.
    이 문단에서만 향수명을 언급할 수 있다.

    추가 규칙:
    - 개별 향수를 장황하게 설명하지 않는다.
    - 부정적인 비교 표현을 사용하지 않는다.
    - 과장된 마케팅 문구를 사용하지 않는다.
    - 문단 사이에는 줄바꿈을 정확히 한 번만 사용한다.
    - 단어 중간에서 줄바꿈하거나 공백을 삽입하지 않는다.
    """

    perfume_block = ""
    for p in perfumes:
        perfume_block += f"""
- {p['perfume_name']} ({p['brand']})
  · 스타일 점수 반영
  · 색상 점수 반영
  · 계절 점수 반영
  · 주요 향조: {p['perfume_mainaccords']}
  · 향의 결: {p['fragrance_desc']}
    · 잘 어울리는 계절: {p['best_seasons']}

  
"""

    user_prompt = f"""
아래는 점수 기반 분석을 통해 선별된 향수 3종의 요약이다.
이 정보를 바탕으로 종합 추천 이유를 작성해줘.

작성 조건:
- 분량은 250~350자
- 개별 향수 설명 ❌
- 공통적인 추천 이유를 중심으로 자세히 설명
- style / color / season 관점에서 왜 함께 묶였는지 설명
- 마지막 문장에서 세 향수의 분위기 차이를 간단히 정리
- 광고 문구, 과장 표현 금지

[사용자 정보]
- 사용자 스타일: {user_style}
- 사용 계절: {user_season}

[추천 향수 요약]
{perfume_block}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.55,
        max_tokens=500
    )

    return response.choices[0].message.content

# 실행 
top3_llm_inputs = build_top3_llm_inputs(
    score_df,
    user_df,
    perfume_df,
    classification_df,
    color_df,
    season_df
)

top3_perfumes = [
    {
        "perfume_name": p["perfume_name"],
        "brand": p["brand"],
        "style_score": p["style_score"],
        "color_score": p["color_score"],
        "season_score": p["season_score"],
        "perfume_mainaccords": p["perfume_mainaccords"],
        "fragrance_desc": p["fragrance_desc"],
        "best_seasons": p["best_seasons"]
    }
    for p in top3_llm_inputs
]
user_style = top3_llm_inputs[0]["user_style"]
user_season = top3_llm_inputs[0]["user_season"]

summary = generate_top3_recommend_summary(
    user_style=user_style,
    user_season=user_season,
    perfumes=top3_perfumes
)

print(summary)
