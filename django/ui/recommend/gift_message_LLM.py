import os
from openai import OpenAI
from dotenv import load_dotenv
from ui.models import UserInfo, Score

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_gift_message_recommendation(user_id, recipient, situation, message_type="짧은"):
    try:
        user = UserInfo.objects.get(user_id=user_id)
        # 추천된 향수들 이름 가져오기 (문구에 녹여내기 위함)
        top3_scores = Score.objects.filter(user=user).select_related('perfume').order_by('-myscore')[:3]
        perfume_names = [s.perfume.perfume_name for s in top3_scores]

        # 시스템 프롬프트 설정 (감성적인 카드 작성자 페르소나)
        system_prompt = f"""
        너는 따뜻하고 감성적인 문장을 쓰는 기프트 카드 작가다. 
        사용자가 '{recipient}'에게 '{situation}'로 향수를 선물하려고 한다.
        선물하는 사람의 마음이 잘 전달되도록 감동적인 카드 문구를 작성하라.

        [작성 규칙]
        1. 말투는 부드럽고 다정하게 한다.
        2. {recipient}와 {situation}의 의미를 문장에 녹여낸다.
        3. 선택된 향수 이미지({", ".join(perfume_names)})의 느낌이 '너의 분위기를 닮았다'는 점을 강조한다.
        4. 결과는 리스트 형태가 아니라, 바로 카드에 적을 수 있는 문장들로만 출력한다.
        """

        if "짧은" in message_type:
            user_prompt = f"1~2문장 내외의 짧고 임팩트 있는 카드 문구 3가지를 추천해줘. 문구 사이에는 '||' 구분자를 넣어줘."
        else:
            user_prompt = f"3~4문장 정도의 진심이 담긴 긴 편지 문구 2가지를 추천해줘. 문구 사이에는 '||' 구분자를 넣어줘."

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,  # 조금 더 창의적인 문구를 위해
            max_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:
        return "선물하는 분의 마음을 담은 예쁜 향기 선물이 되길 바라요.||이 향기가 너의 하루를 더 특별하게 만들어줄 거야."