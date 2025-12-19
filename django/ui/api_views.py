import os
import random
from urllib.parse import quote

from django.db import transaction
from django.conf import settings
from django.templatetags.static import static
from django.utils.safestring import mark_safe

# DRF(Django REST Framework) 관련 임포트
from rest_framework.views import APIView
from rest_framework import viewsets, filters, status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.renderers import BrowsableAPIRenderer, JSONRenderer

# 모델 및 시리얼라이저 임포트
from .models import (
    TopBottom, Dress, ClothesColor, PerfumeColor,
    Perfume, PerfumeSeason, PerfumeClassification, UserInfo, Score
)
from .serializers import (
    TopBottomSerializer,
    DressSerializer,
    ClothesColorSerializer,
    PerfumeColorSerializer,
    PerfumeSeasonSerializer,
    PerfumeSerializer,
    PerfumeClassificationSerializer,
    UserInputSerializer
)

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import UserInputSerializer
from ui.models import Score, Perfume, TopBottom, Dress
from ui.recommend.calculation import get_user_data, recommend_perfumes
from django.db import transaction
from rest_framework.renderers import JSONRenderer

# =============================================================
# 1. 이미지 데이터 조회 API (JSON + HTML 미리보기 기능 포함)
# =============================================================
class FilterImagesAPI(APIView):
    renderer_classes = [BrowsableAPIRenderer, JSONRenderer]

    def get_view_description(self, html=False):
        """
        화면 상단 설명창 (이미지 미리보기 포함)
        """
        description = "<h3>📸 이미지 조회 결과 (미리보기)</h3><p>아래 회색 박스는 데이터(JSON)이고, 실제 이미지는 여기에 나옵니다.</p>"

        request = self.request
        category_en = request.query_params.get('category')
        item_en = request.query_params.get('item')
        color_en = request.query_params.get('color')

        # 매핑
        map_category = {'top': '상의', 'bottom': '하의', 'onepiece': '원피스'}
        map_item = {'blouse': '블라우스', 'tshirt': '티셔츠', 'knit': '니트웨어', 'shirt': '셔츠', 'sleeveless': '탑',
                    'hoodie': '후드티', 'sweatshirt': '맨투맨', 'bratop': '브라탑', 'pants': '팬츠', 'jeans': '청바지',
                    'skirt': '스커트', 'long_skirt': '롱스커트', 'leggings': '레깅스', 'jogger': '트레이닝', 'slacks': '슬랙스',
                    'dress': '드레스', 'onepiece': '원피스', 'jumpsuit': '점프수트'}
        map_color = {'white': '화이트', 'black': '블랙', 'grey': '그레이', 'charcoal': '차콜', 'beige': '베이지', 'ivory': '아이보리',
                     'brown': '브라운', 'camel': '카멜', 'navy': '네이비', 'blue': '블루', 'skyblue': '스카이블루', 'jeans_blue': '진청',
                     'light_blue': '연청', 'middle_blue': '중청', 'red': '레드', 'pink': '핑크', 'wine': '와인', 'rose': '로즈',
                     'purple': '퍼플', 'lavender': '라벤더', 'violet': '바이올렛', 'yellow': '옐로우', 'mustard': '머스타드',
                     'orange': '오렌지', 'green': '그린', 'khaki': '카키', 'mint': '민트', 'olive': '올리브', 'neon': '네온',
                     'gold': '골드', 'silver': '실버', 'pattern': '패턴', 'unknown': 'unknown'}

        cat_kr = map_category.get(category_en)
        item_kr = map_item.get(item_en)
        color_kr = map_color.get(color_en)

        img_html = "<div style='display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px;'>"

        if cat_kr and item_kr and color_kr:
            base_dir = os.path.join(settings.BASE_DIR, 'ui', 'static', 'ui', 'clothes', cat_kr, item_kr, color_kr)
            valid_images = []
            if os.path.exists(base_dir):
                try:
                    files = os.listdir(base_dir)
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                            # [수정됨] quote() 제거! 한글 그대로 사용
                            url_path = f'/static/ui/clothes/{cat_kr}/{item_kr}/{color_kr}/{file}'
                            valid_images.append(url_path)
                except:
                    pass

            count = min(len(valid_images), 4)
            selected = random.sample(valid_images, count) if valid_images else []

            for img in selected:
                img_html += f"<img src='{img}' style='width: 150px; height: 150px; object-fit: cover; border-radius: 8px; border: 1px solid #ddd;'>"

        img_html += "</div>"
        return mark_safe(description + img_html)

    def get(self, request):
        """
        JSON 응답 반환
        """
        category_en = request.query_params.get('category')
        item_en = request.query_params.get('item')
        color_en = request.query_params.get('color')

        if not (category_en and item_en and color_en):
            return Response({'images': []})

        # 매핑
        map_category = {'top': '상의', 'bottom': '하의', 'onepiece': '원피스'}
        map_item = {'blouse': '블라우스', 'tshirt': '티셔츠', 'knit': '니트웨어', 'shirt': '셔츠', 'sleeveless': '탑',
                    'hoodie': '후드티', 'sweatshirt': '맨투맨', 'bratop': '브라탑', 'pants': '팬츠', 'jeans': '청바지',
                    'skirt': '스커트', 'long_skirt': '롱스커트', 'leggings': '레깅스', 'jogger': '트레이닝', 'slacks': '슬랙스',
                    'dress': '드레스', 'onepiece': '원피스', 'jumpsuit': '점프수트'}
        map_color = {'white': '화이트', 'black': '블랙', 'grey': '그레이', 'charcoal': '차콜', 'beige': '베이지', 'ivory': '아이보리',
                     'brown': '브라운', 'camel': '카멜', 'navy': '네이비', 'blue': '블루', 'skyblue': '스카이블루', 'jeans_blue': '진청',
                     'light_blue': '연청', 'middle_blue': '중청', 'red': '레드', 'pink': '핑크', 'wine': '와인', 'rose': '로즈',
                     'purple': '퍼플', 'lavender': '라벤더', 'violet': '바이올렛', 'yellow': '옐로우', 'mustard': '머스타드',
                     'orange': '오렌지', 'green': '그린', 'khaki': '카키', 'mint': '민트', 'olive': '올리브', 'neon': '네온',
                     'gold': '골드', 'silver': '실버', 'pattern': '패턴', 'unknown': 'unknown'}

        cat_kr = map_category.get(category_en)
        item_kr = map_item.get(item_en)
        color_kr = map_color.get(color_en)

        if not (cat_kr and item_kr and color_kr):
            return Response({'images': []})

        base_dir = os.path.join(settings.BASE_DIR, 'ui', 'static', 'ui', 'clothes', cat_kr, item_kr, color_kr)
        valid_images = []

        if os.path.exists(base_dir):
            try:
                files = os.listdir(base_dir)
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        url_path = f'/static/ui/clothes/{cat_kr}/{item_kr}/{color_kr}/{file}'
                        valid_images.append(url_path)
            except:
                pass

        count = min(len(valid_images), 4)
        selected_images = random.sample(valid_images, count) if valid_images else []
        while len(selected_images) < 4:
            selected_images.append(None)

        return Response({'images': selected_images})
# =============================================================
# 2. 향수 목록 조회 API (검색 기능 추가됨)
# =============================================================
class PerfumeViewSet(viewsets.ModelViewSet):
    """
    [기능]
    1. 전체 향수 목록 조회
    2. 검색 기능 (?search=Chanel 또는 ?search=No.5)
    """
    queryset = Perfume.objects.all().order_by('perfume_id')
    serializer_class = PerfumeSerializer

    # 검색 필터 장착
    filter_backends = [filters.SearchFilter]
    # 브랜드명과 향수명으로 검색 가능
    search_fields = ['brand', 'perfume_name']


# =============================================================
# 3. 기타 데이터 관리 ViewSets (기본 CRUD)
# =============================================================

class ClothesColorViewSet(viewsets.ModelViewSet):
    queryset = ClothesColor.objects.all()
    serializer_class = ClothesColorSerializer


class PerfumeColorViewSet(viewsets.ModelViewSet):
    queryset = PerfumeColor.objects.all()
    serializer_class = PerfumeColorSerializer


class TopBottomViewSet(viewsets.ModelViewSet):
    queryset = TopBottom.objects.all()
    serializer_class = TopBottomSerializer


class DressViewSet(viewsets.ModelViewSet):
    queryset = Dress.objects.all()
    serializer_class = DressSerializer


class PerfumeSeasonViewSet(viewsets.ModelViewSet):
    queryset = PerfumeSeason.objects.all()
    serializer_class = PerfumeSeasonSerializer


class PerfumeClassificationViewSet(viewsets.ModelViewSet):
    queryset = PerfumeClassification.objects.all()
    serializer_class = PerfumeClassificationSerializer


# ui/api_views.py

class UserInputView(APIView):
    def post(self, request):
        serializer = UserInputSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data

        try:
            # ===================================================
            # 1. 매핑 준비
            # ===================================================

            # [옷 종류 매핑] (영어 -> 한글)
            map_item = {
                'blouse': '블라우스', 'tshirt': '티셔츠', 'knit': '니트웨어', 'shirt': '셔츠', 'sleeveless': '탑',
                'hoodie': '후드티', 'sweatshirt': '맨투맨', 'bratop': '브라탑',
                'pants': '팬츠', 'jeans': '청바지', 'skirt': '스커트', 'long_skirt': '롱스커트', 'leggings': '레깅스',
                'jogger': '트레이닝', 'slacks': '슬랙스',
                'dress': '드레스', 'onepiece': '원피스', 'jumpsuit': '점프수트'
            }


            map_color = {
                'white': '화이트',
                'black': '블랙',
                'beige': '베이지',
                'pink': '핑크',
                'skyblue': '스카이블루',
                'grey': '그레이',
                'brown': '브라운',
                'navy': '네이비',
                'red': '레드',
                'yellow': '옐로우',
                'blue': '블루',
                'lavender': '라벤더',
                'wine': '와인',
                'silver': '실버',
                'orange': '오렌지',
                'khaki': '카키',
                'green': '그린',
                'purple': '퍼플',
                'mint': '민트',
                'gold': '골드',
                'neon': '네온',
                'jeans_blue': '진청'  # 프론트엔드 코드에 있어서 유지함
            }

            # ===================================================
            # 2. 데이터 변환
            # ===================================================

            # [수정됨] 계절은 영어 그대로 사용 (매핑 X)
            final_season = data['season']

            # [수정됨] 향조도 영어 그대로 사용 (매핑 X), 리스트만 문자열로 변환
            dislikes_list = data.get('disliked_accords', [])
            dislikes_str = ", ".join(dislikes_list) if dislikes_list else None

            # 옷/색상은 한글로 변환 (DB 텍스트 저장용)
            top_kr = map_item.get(data.get('top'))
            top_color_kr = map_color.get(data.get('top_color'))

            bottom_kr = map_item.get(data.get('bottom'))
            bottom_color_kr = map_color.get(data.get('bottom_color'))

            onepiece_kr = map_item.get(data.get('onepiece'))
            onepiece_color_kr = map_color.get(data.get('onepiece_color'))

            # ===================================================
            # 3. ID 찾기 (FK 연결용 객체 생성)
            # ===================================================
            # 주의: FK 연결할 때 ClothesColor 테이블은 '영어 키(white)'를 사용할 수도 있으므로
            # data['top_color'] (영어)를 그대로 사용합니다.

            user_top_obj = None
            user_bottom_obj = None
            user_dress_obj = None

            # [CASE A] 상의 + 하의
            if data.get('top') and data.get('bottom'):
                # 1. 색상 객체 (영어 키 사용)
                top_color_obj, _ = ClothesColor.objects.get_or_create(color=data['top_color'])
                # 2. 상의 객체 (카테고리: 영어, 색상: 객체)
                user_top_obj, _ = TopBottom.objects.get_or_create(
                    top_category=data['top'],
                    top_color=top_color_obj,
                    defaults={'style': 'basic'}
                )

                bottom_color_obj, _ = ClothesColor.objects.get_or_create(color=data['bottom_color'])
                user_bottom_obj, _ = TopBottom.objects.get_or_create(
                    bottom_category=data['bottom'],
                    bottom_color=bottom_color_obj,
                    defaults={'style': 'basic'}
                )

            # [CASE B] 원피스
            elif data.get('onepiece'):
                dress_color_obj, _ = ClothesColor.objects.get_or_create(color=data['onepiece_color'])
                user_dress_obj, _ = Dress.objects.get_or_create(
                    sub_style=data['onepiece'],
                    dress_color=dress_color_obj,
                    defaults={'style': 'basic'}
                )

            # ===================================================
            # 4. UserInfo 저장
            # ===================================================
            UserInfo.objects.all().delete()

            new_user_info = UserInfo.objects.create(
                season=final_season,  # 영어 (spring)
                disliked_accord=dislikes_str,  # 영어 (citrus, woody)

                # ID 연결 (Foreign Key)
                top_id=user_top_obj,
                bottom_id=user_bottom_obj,
                dress_id=user_dress_obj,

                # 텍스트 정보 저장 (한글) - 사진 목록에 맞춘 값
                top_category=top_kr,
                top_color=top_color_kr,  # 예: 화이트, 블랙...
                bottom_category=bottom_kr,
                bottom_color=bottom_color_kr,
                dress_color=onepiece_color_kr
            )

            return Response(
                {"message": "저장 성공!", "user_id": new_user_info.user_id},
                status=status.HTTP_201_CREATED
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response(
                {"error": str(e), "type": type(e).__name__},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# 2) 추천 알고리즘 점수 계산 및 score 테이블 저장 api
class RecommendationView(APIView):
    renderer_classes = [JSONRenderer]

    def get(self, request):
        user_id = request.query_params.get("user_id")
        # ... (중략: user_id 체크 로직) ...

        try:
            data = get_user_data(user_id)

            # 중요: recommend_perfumes 호출 시 인자 이름을 calculation.py의 정의와 일치시킴
            results = recommend_perfumes(
                user_info=[data],
                perfume=data["perfumes"],  # get_user_data에서 만든 리스트
                perfume_classification=list(PerfumeClassification.objects.all().values("perfume_id", "fragrance")),
                perfume_season=list(
                    PerfumeSeason.objects.all().values("perfume_id", "spring", "summer", "fall", "winter")),
                상의_하의=list(TopBottom.objects.all().values()),
                원피스=list(Dress.objects.all().values()),
                clothes_color=data["clothes_color"],
                perfume_color=data["perfume_color"],
            )

            print(f"DEBUG: 계산된 결과 개수 = {len(results)}")  # 터미널 확인용

            if not results:
                return Response({"message": "추천 결과가 없습니다."}, status=200)

            # 기존 데이터 먼저 삭제
            Score.objects.all().delete()

            # 결과 저장 (update_or_create 사용)
            with transaction.atomic():
                for res in results:
                    Score.objects.update_or_create(
                        perfume_id=res["perfume_id"],  # FK 객체 직접 할당 또는 ID
                        defaults={
                            "season_score": res["season_score"],
                            "color_score": res["color_score"],
                            "style_score": res["style_score"],
                            "myscore": res["myscore"]
                        }
                    )

            return Response({"results": results}, status=status.HTTP_201_CREATED)

        except Exception as e:
            import traceback
            traceback.print_exc()  # 에러가 나면 터미널에 상세 내용을 찍음
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)