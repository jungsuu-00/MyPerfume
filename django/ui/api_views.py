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
from .serializers import UserInputSerializer, RecommendationResultSerializer
from ui.models import Score, Perfume, TopBottom, Dress
from ui.recommend.calculation import get_user_data, recommend_perfumes
from django.db import transaction
from rest_framework.renderers import JSONRenderer

# =============================================================
# 1. 이미지 데이터 조회 API
# =============================================================
class FilterImagesAPI(APIView):
    renderer_classes = [JSONRenderer]

    def get(self, request):
        category_en = request.query_params.get('category')
        item_en = request.query_params.get('item')
        color_en = request.query_params.get('color')

        if not (category_en and item_en and color_en):
            return Response({'images': []})

        # [누락 없는 매핑]
        map_category = {'top': '상의', 'bottom': '하의', 'onepiece': '원피스'}
        map_item = {
            'blouse': '블라우스', 'tshirt': '티셔츠', 'knit': '니트웨어', 'shirt': '셔츠', 'hoodie': '후드티',
            'pants': '팬츠', 'jeans': '청바지', 'skirt': '스커트', 'leggings': '레깅스',
            'dress': '드레스', 'jumpsuit': '점프수트'
        }
        map_color = {
            'white': '화이트', 'black': '블랙', 'grey': '그레이', 'navy': '네이비', 'beige': '베이지',
            'pink': '핑크', 'skyblue': '스카이블루', 'brown': '브라운', 'red': '레드', 'green': '그린',
            'gold': '골드', 'silver': '실버'
        }

        cat_kr = map_category.get(category_en)
        item_kr = map_item.get(item_en)
        color_kr = map_color.get(color_en)

        if not (cat_kr and item_kr and color_kr):
            return Response({'images': []})

        # 실제 서버 내 폴더 경로 (한글 그대로 사용)
        base_dir = os.path.join(settings.BASE_DIR, 'ui', 'static', 'ui', 'clothes', cat_kr, item_kr, color_kr)
        valid_images = []

        if os.path.exists(base_dir):
            try:
                files = os.listdir(base_dir)
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        # [중요] 브라우저용 URL은 한글 부분을 반드시 quote로 인코딩해야 함
                        encoded_cat = quote(cat_kr)
                        encoded_item = quote(item_kr)
                        encoded_color = quote(color_kr)
                        encoded_file = quote(file)

                        url_path = f'/static/ui/clothes/{encoded_cat}/{encoded_item}/{encoded_color}/{encoded_file}'
                        valid_images.append(url_path)
            except Exception as e:
                print(f"Error reading directory: {e}")

        # 무작위 4개 선택
        selected_images = random.sample(valid_images, min(len(valid_images), 4)) if valid_images else []
        # 부족한 경우 null로 채움 (프론트엔드 형식 유지)
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
            # ... (중략: map_item, map_color 매핑 로직은 동일) ...
            map_item = {
                'blouse': '블라우스', 'tshirt': '티셔츠', 'knit': '니트웨어', 'shirt': '셔츠', 'sleeveless': '탑',
                'hoodie': '후드티', 'sweatshirt': '맨투맨', 'bratop': '브라탑',
                'pants': '팬츠', 'jeans': '청바지', 'skirt': '스커트', 'long_skirt': '롱스커트', 'leggings': '레깅스',
                'jogger': '트레이닝', 'slacks': '슬랙스',
                'dress': '드레스', 'onepiece': '원피스', 'jumpsuit': '점프수트'
            }
            map_color = {
                'white': '화이트', 'black': '블랙', 'beige': '베이지', 'pink': '핑크',
                'skyblue': '스카이블루', 'grey': '그레이', 'brown': '브라운', 'navy': '네이비',
                'red': '레드', 'yellow': '옐로우', 'blue': '블루', 'lavender': '라벤더',
                'wine': '와인', 'silver': '실버', 'orange': '오렌지', 'khaki': '카키',
                'green': '그린', 'purple': '퍼플', 'mint': '민트', 'gold': '골드',
                'neon': '네온', 'jeans_blue': '진청'
            }

            final_season = data['season']
            dislikes_list = data.get('disliked_accords', [])
            dislikes_str = ", ".join(dislikes_list) if dislikes_list else None

            # [핵심 수정] 시리얼라이저에서 넘겨준 이미지 경로 꺼내오기
            top_img_url = data.get('top_img')
            bottom_img_url = data.get('bottom_img')
            onepiece_img_url = data.get('onepiece_img')

            # ... (중략: FK 객체 찾는 로직 동일) ...
            user_top_obj = None
            user_bottom_obj = None
            user_dress_obj = None

            if data.get('top') and data.get('bottom'):
                top_color_obj, _ = ClothesColor.objects.get_or_create(color=data['top_color'])
                user_top_obj, _ = TopBottom.objects.get_or_create(top_category=data['top'], top_color=top_color_obj, defaults={'style': 'basic'})
                bottom_color_obj, _ = ClothesColor.objects.get_or_create(color=data['bottom_color'])
                user_bottom_obj, _ = TopBottom.objects.get_or_create(bottom_category=data['bottom'], bottom_color=bottom_color_obj, defaults={'style': 'basic'})
            elif data.get('onepiece'):
                dress_color_obj, _ = ClothesColor.objects.get_or_create(color=data['onepiece_color'])
                user_dress_obj, _ = Dress.objects.get_or_create(sub_style=data['onepiece'], dress_color=dress_color_obj, defaults={'style': 'basic'})

            # [핵심 수정] UserInfo 저장 시 이미지 필드 명시
            new_user_info = UserInfo.objects.create(
                season=final_season,
                disliked_accord=dislikes_str,
                top_id=user_top_obj,
                bottom_id=user_bottom_obj,
                dress_id=user_dress_obj,

                # 아래 이미지 경로 저장 부분이 누락되어 있었습니다!
                top_img=top_img_url,
                bottom_img=bottom_img_url,
                dress_img=onepiece_img_url,

                top_category=map_item.get(data.get('top')),
                top_color=map_color.get(data.get('top_color')),
                bottom_category=map_item.get(data.get('bottom')),
                bottom_color=map_color.get(data.get('bottom_color')),
                dress_color=map_color.get(data.get('onepiece_color'))
            )

            return Response({"message": "저장 성공!", "user_id": new_user_info.user_id}, status=status.HTTP_201_CREATED)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class UserOutfitAPIView(APIView):
    """
    사용자가 방금 선택한 코디 이미지 경로만 반환하는 전용 API
    """
    renderer_classes = [JSONRenderer]

    def get(self, request):
        # 가장 최근에 저장된 사용자 정보 가져오기
        last_user = UserInfo.objects.last()

        if not last_user:
            return Response({"error": "데이터가 없습니다."}, status=status.HTTP_404_NOT_FOUND)

        # 이미지 경로 데이터만 구성
        data = {
            "top_img": last_user.top_img,
            "bottom_img": last_user.bottom_img,
            "onepiece_img": last_user.dress_img,  # 모델 필드명 확인 필요
        }
        return Response(data, status=status.HTTP_200_OK)


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


# api_views.py 내 RecommendationResultAPIView 수정

class RecommendationResultAPIView(APIView):
    renderer_classes = [JSONRenderer]

    def get(self, request):
        # 1. 점수와 상관없이 가장 최근 사용자 정보는 무조건 가져옴
        last_user = UserInfo.objects.last()

        # 2. 점수 결과 가져오기 (고장 났더라도 에러 내지 않음)
        results = Score.objects.all().select_related(
            'perfume', 'perfume__season', 'perfume__mainaccord1', 'perfume__mainaccord2', 'perfume__mainaccord3'
        ).order_by('-myscore')

        # 3. 향수 데이터 시리얼라이징 (결과가 있으면 변환, 없으면 빈 리스트)
        perfumes_data = []
        if results.exists():
            perfume_serializer = RecommendationResultSerializer(results, many=True)
            perfumes_data = perfume_serializer.data

        # 4. 최종 응답 (상태 코드 200으로 고정하여 자바스크립트가 멈추지 않게 함)
        response_data = {
            "user_outfit": {
                "top_img": last_user.top_img if last_user else None,
                "bottom_img": last_user.bottom_img if last_user else None,
                "onepiece_img": last_user.dress_img if last_user else None,
            },
            "perfumes": perfumes_data  # 점수 고장 시 빈 배열 [] 이 감
        }
        return Response(response_data, status=status.HTTP_200_OK)