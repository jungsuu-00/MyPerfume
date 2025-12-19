from rest_framework import serializers
from .models import TopBottom, Dress, ClothesColor, PerfumeColor, Perfume, PerfumeSeason, PerfumeClassification, UserInfo


# ==========================================
# 1. 데이터 관리용 Serializers
# ==========================================

# 1. 옷 색상
class ClothesColorSerializer(serializers.ModelSerializer):
    class Meta:
        model = ClothesColor
        fields = '__all__'


# 2. 향수 컬러(향조)
class PerfumeColorSerializer(serializers.ModelSerializer):
    class Meta:
        model = PerfumeColor
        fields = '__all__'


# 3. 상의 & 하의
class TopBottomSerializer(serializers.ModelSerializer):
    class Meta:
        model = TopBottom
        fields = '__all__'


# 4. 원피스
class DressSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dress
        fields = '__all__'

# 5. 향수
class PerfumeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Perfume
        fields = '__all__'

# 6. 계절
class PerfumeSeasonSerializer(serializers.ModelSerializer):
    class Meta:
        model = PerfumeSeason
        fields = '__all__'

#7. 향수 분류 정보

class PerfumeClassificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = PerfumeClassification
        fields = '__all__'


class UserInputSerializer(serializers.Serializer):
    """
    프론트엔드에서 보내주는 JSON 데이터를 검증하는 시리얼라이저
    (DB 모델과 직접 연결하지 않고, 입력값 검증에 집중합니다)
    """
    # 1. 공통 정보
    season = serializers.CharField(required=True)
    disliked_accords = serializers.ListField(
        child=serializers.CharField(), required=False, allow_empty=True
    )

    # 2. 상의/하의 정보 (선택 사항)
    top = serializers.CharField(required=False, allow_null=True)
    top_color = serializers.CharField(required=False, allow_null=True)
    bottom = serializers.CharField(required=False, allow_null=True)
    bottom_color = serializers.CharField(required=False, allow_null=True)

    # 3. 원피스 정보 (선택 사항)
    onepiece = serializers.CharField(required=False, allow_null=True)
    onepiece_color = serializers.CharField(required=False, allow_null=True)

    def validate(self, data):
        """
        데이터 유효성 검사 (필수 조합 확인)
        """
        top = data.get("top")
        bottom = data.get("bottom")
        onepiece = data.get("onepiece")

        season = data.get("season")

        # 1. 계절 필수 확인
        if not season:
            raise serializers.ValidationError("계절은 필수 항목입니다.")

        # 2. 옷 조합 확인 (투피스 OR 원피스)
        has_2pc = top and data.get("top_color") and bottom and data.get("bottom_color")
        has_1pc = onepiece and data.get("onepiece_color")

        if not (has_2pc or has_1pc):
            raise serializers.ValidationError("코디 정보(상의+하의 또는 원피스)를 모두 선택해야 합니다.")

        return data
