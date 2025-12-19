# from django.db import models
#
#
# # ==========================================
# # 1. 참조용 기초 테이블 (색상, 어코드)
# # ==========================================
#
# class ClothesColor(models.Model):
#     # Table: clothes_color
#     # color varchar(100) [pk]
#     color = models.CharField(max_length=100, primary_key=True, db_column="color")
#     rgb_tuple = models.CharField(max_length=100, null=True, blank=True, db_column="rgb_tuple")
#
#     def __str__(self):
#         return self.color
#
#     class Meta:
#         db_table = "clothes_color"
#         app_label = 'ui'
#
#
# class PerfumeColor(models.Model):
#     # Table: perfume_color
#     # mainaccord varchar [primary key]
#     mainaccord = models.CharField(max_length=255, primary_key=True, db_column="mainaccord")
#     color = models.CharField(max_length=255, null=True, blank=True, db_column="color")
#
#     def __str__(self):
#         return self.mainaccord
#
#     class Meta:
#         db_table = "perfume_color"
#         app_label = 'ui'
#
#
# # ==========================================
# # 2. 옷 데이터 (상의_하의, 원피스)
# # ==========================================
#
# class TopBottom(models.Model):
#     # Table: 상의_하의
#     # 식별자 integer [primary key]
#     id = models.AutoField(primary_key=True, db_column="식별자")
#
#     style = models.CharField(max_length=100, null=True, blank=True, db_column="스타일")
#     sub_style = models.CharField(max_length=100, null=True, blank=True, db_column="서브스타일")
#
#     # [FK] 상의_색상 -> clothes_color.color
#     top_color = models.ForeignKey(ClothesColor, on_delete=models.PROTECT, related_name="top_set", null=True, blank=True,
#                                   db_column="상의_색상")
#     top_category = models.CharField(max_length=100, null=True, blank=True, db_column="상의_카테고리")
#     top_sleeve_length = models.CharField(max_length=100, null=True, blank=True, db_column="상의_소매기장")
#     top_material = models.CharField(max_length=100, null=True, blank=True, db_column="상의_소재")
#     top_print = models.CharField(max_length=100, null=True, blank=True, db_column="상의_프린트")
#     top_neckline = models.CharField(max_length=100, null=True, blank=True, db_column="상의_넥라인")
#     top_fit = models.CharField(max_length=100, null=True, blank=True, db_column="상의_핏")
#
#     bottom_length = models.CharField(max_length=100, null=True, blank=True, db_column="하의_기장")
#     # [FK] 하의_색상 -> clothes_color.color
#     bottom_color = models.ForeignKey(ClothesColor, on_delete=models.PROTECT, related_name="bottom_set", null=True,
#                                      blank=True, db_column="하의_색상")
#     bottom_category = models.CharField(max_length=100, null=True, blank=True, db_column="하의_카테고리")
#     bottom_material = models.CharField(max_length=100, null=True, blank=True, db_column="하의_소재")
#     bottom_fit = models.CharField(max_length=100, null=True, blank=True, db_column="하의_핏")
#
#
#     class Meta:
#         db_table = "상의_하의"
#         app_label = 'ui'
#
#
# class Dress(models.Model):
#     # Table: 원피스
#     # 식별자 int [pk]
#     id = models.AutoField(primary_key=True, db_column="식별자")
#
#     style = models.CharField(max_length=100, null=True, blank=True, db_column="스타일")
#     sub_style = models.CharField(max_length=100, null=True, blank=True, db_column="서브스타일")
#
#     dress_length = models.CharField(max_length=100, null=True, blank=True, db_column="원피스_기장")
#     # [FK] 원피스_색상 -> clothes_color.color
#     dress_color = models.ForeignKey(ClothesColor, on_delete=models.PROTECT, related_name="dress_set", null=True,
#                                     blank=True, db_column="원피스_색상")
#     dress_sleeve_length = models.CharField(max_length=100, null=True, blank=True, db_column="원피스_소매기장")
#     dress_material = models.CharField(max_length=100, null=True, blank=True, db_column="원피스_소재")
#     dress_print = models.CharField(max_length=100, null=True, blank=True, db_column="원피스_프린트")
#     dress_fit = models.CharField(max_length=100, null=True, blank=True, db_column="원피스_핏")
#     dress_detail = models.CharField(max_length=255, null=True, blank=True, db_column="원피스_디테일")
#
#     class Meta:
#         db_table = "원피스"
#         app_label = 'ui'
#
#
# class UserInfo(models.Model):
#     # Table: user_info
#     # 사용자_식별자 integer [primary key]
#     user_id = models.AutoField(primary_key=True, db_column="사용자_식별자")
#
#     # [FK] 상의/하의/원피스 식별자 연결
#     top_id = models.ForeignKey(TopBottom, on_delete=models.SET_NULL, related_name="user_tops", null=True, blank=True,
#                                db_column="상의_식별자")
#     bottom_id = models.ForeignKey(TopBottom, on_delete=models.SET_NULL, related_name="user_bottoms", null=True,
#                                   blank=True, db_column="하의_식별자")
#     dress_id = models.ForeignKey(Dress, on_delete=models.SET_NULL, related_name="user_dresses", null=True, blank=True,
#                                  db_column="원피스_식별자")
#
#     season = models.CharField(max_length=50, null=True, blank=True, db_column="계절")
#     disliked_accord = models.CharField(max_length=50, null=True, blank=True, db_column="비선호_향조")
#
#     class Meta:
#         db_table = "user_info"
#         app_label = 'ui'
#
#
# # ==========================================
# # 3. 향수 데이터 (Perfume)
# # ==========================================
#
# class Perfume(models.Model):
#     # Table: perfume
#     # perfume_id int [pk]
#     perfume_id = models.IntegerField(primary_key=True,
#                                      db_column="perfume_id")  # AutoField 대신 CSV ID 유지를 위해 IntegerField 추천
#
#     url = models.CharField(max_length=500, null=True, blank=True, db_column="url")
#     perfume_name = models.CharField(max_length=100, db_column="Perfume")
#     brand = models.CharField(max_length=50, db_column="Brand")
#     country = models.CharField(max_length=50, null=True, blank=True, db_column="Country")
#     gender = models.CharField(max_length=20, null=True, blank=True, db_column="Gender")
#
#     rating_value = models.FloatField(default=0.0, db_column="RatingValue")
#     rating_count = models.IntegerField(default=0, db_column="RatingCount")
#     year = models.IntegerField(null=True, blank=True, db_column="Year")
#
#     top = models.TextField(null=True, blank=True, db_column="Top")
#     middle = models.TextField(null=True, blank=True, db_column="Middle")
#     base = models.TextField(null=True, blank=True, db_column="Base")
#
#     # [FK] Main Accords -> perfume_color.mainaccord
#     mainaccord1 = models.ForeignKey(PerfumeColor, on_delete=models.SET_NULL, related_name="p_accord1", null=True,
#                                     blank=True, db_column="mainaccord1")
#     mainaccord2 = models.ForeignKey(PerfumeColor, on_delete=models.SET_NULL, related_name="p_accord2", null=True,
#                                     blank=True, db_column="mainaccord2")
#     mainaccord3 = models.ForeignKey(PerfumeColor, on_delete=models.SET_NULL, related_name="p_accord3", null=True,
#                                     blank=True, db_column="mainaccord3")
#     mainaccord4 = models.ForeignKey(PerfumeColor, on_delete=models.SET_NULL, related_name="p_accord4", null=True,
#                                     blank=True, db_column="mainaccord4")
#     mainaccord5 = models.ForeignKey(PerfumeColor, on_delete=models.SET_NULL, related_name="p_accord5", null=True,
#                                     blank=True, db_column="mainaccord5")
#
#     def __str__(self):
#         return f"{self.brand} - {self.perfume_name}"
#
#     class Meta:
#         db_table = "perfume"
#         app_label = 'ui'
#
#
# class PerfumeSeason(models.Model):
#     # Table: perfume_season
#     # perfume_id integer [primary key] -> OneToOne with Perfume
#     perfume = models.OneToOneField(Perfume, primary_key=True, on_delete=models.CASCADE, related_name="season",
#                                    db_column="perfume_id")
#     spring = models.FloatField(default=0.0, db_column="spring")
#     summer = models.FloatField(default=0.0, db_column="summer")
#     fall = models.FloatField(default=0.0, db_column="fall")
#     winter = models.FloatField(default=0.0, db_column="winter")
#
#     class Meta:
#         db_table = "perfume_season"
#         app_label = 'ui'
#
#
# class PerfumeClassification(models.Model):
#     # Table: perfume_classification
#     # perfume_id integer [primary key] -> OneToOne with Perfume
#     perfume = models.OneToOneField(Perfume, primary_key=True, on_delete=models.CASCADE, related_name="classification",
#                                    db_column="perfume_id")
#     fragrance = models.CharField(max_length=255, null=True, blank=True, db_column="fragrance")
#
#     class Meta:
#         db_table = "perfume_classification"
#         app_label = 'ui'
#
#
# class Score(models.Model):
#     # Table: score
#     # perfume_id integer [primary key] -> OneToOne with Perfume
#     perfume = models.OneToOneField(Perfume, primary_key=True, on_delete=models.CASCADE, related_name="score",
#                                    db_column="perfume_id")
#     season_score = models.FloatField(default=0.0, db_column="season_score")
#     color_score = models.FloatField(default=0.0, db_column="color_score")
#     style_score = models.FloatField(default=0.0, db_column="style_score")
#     myscore = models.FloatField(default=0.0, db_column="myscore")
#
#     class Meta:
#         db_table = "score"
#         app_label = 'ui'

from django.db import models


# ==========================================
# 1. 참조용 기초 테이블 (색상, 어코드)
# ==========================================

class ClothesColor(models.Model):
    # Table: clothes_color
    # color varchar(100) [pk]
    color = models.CharField(max_length=100, primary_key=True, db_column="color")
    rgb_tuple = models.CharField(max_length=100, null=True, blank=True, db_column="rgb_tuple")

    def __str__(self):
        return self.color

    class Meta:
        db_table = "clothes_color"
        app_label = 'ui'


class PerfumeColor(models.Model):
    # Table: perfume_color
    # mainaccord varchar [primary key]
    mainaccord = models.CharField(max_length=255, primary_key=True, db_column="mainaccord")
    color = models.CharField(max_length=255, null=True, blank=True, db_column="color")

    def __str__(self):
        return self.mainaccord

    class Meta:
        db_table = "perfume_color"
        app_label = 'ui'


# ==========================================
# 2. 옷 데이터 (상의_하의, 원피스)
# ==========================================

class TopBottom(models.Model):
    # Table: 상의_하의
    # 식별자 integer [primary key]
    id = models.AutoField(primary_key=True, db_column="식별자")

    style = models.CharField(max_length=100, null=True, blank=True, db_column="스타일")
    sub_style = models.CharField(max_length=100, null=True, blank=True, db_column="서브스타일")

    # [FK] 상의_색상 -> clothes_color.color
    top_color = models.ForeignKey(ClothesColor, on_delete=models.PROTECT, related_name="top_set", null=True, blank=True,
                                  db_column="상의_색상")
    top_category = models.CharField(max_length=100, null=True, blank=True, db_column="상의_카테고리")
    top_sleeve_length = models.CharField(max_length=100, null=True, blank=True, db_column="상의_소매기장")
    top_material = models.CharField(max_length=100, null=True, blank=True, db_column="상의_소재")
    top_print = models.CharField(max_length=100, null=True, blank=True, db_column="상의_프린트")
    top_neckline = models.CharField(max_length=100, null=True, blank=True, db_column="상의_넥라인")
    top_fit = models.CharField(max_length=100, null=True, blank=True, db_column="상의_핏")

    bottom_length = models.CharField(max_length=100, null=True, blank=True, db_column="하의_기장")
    # [FK] 하의_색상 -> clothes_color.color
    bottom_color = models.ForeignKey(ClothesColor, on_delete=models.PROTECT, related_name="bottom_set", null=True,
                                     blank=True, db_column="하의_색상")
    bottom_category = models.CharField(max_length=100, null=True, blank=True, db_column="하의_카테고리")
    bottom_material = models.CharField(max_length=100, null=True, blank=True, db_column="하의_소재")
    bottom_fit = models.CharField(max_length=100, null=True, blank=True, db_column="하의_핏")


    class Meta:
        db_table = "상의_하의"
        app_label = 'ui'


class Dress(models.Model):
    # Table: 원피스
    # 식별자 int [pk]
    id = models.AutoField(primary_key=True, db_column="식별자")

    style = models.CharField(max_length=100, null=True, blank=True, db_column="스타일")
    sub_style = models.CharField(max_length=100, null=True, blank=True, db_column="서브스타일")

    dress_length = models.CharField(max_length=100, null=True, blank=True, db_column="원피스_기장")
    # [FK] 원피스_색상 -> clothes_color.color
    dress_color = models.ForeignKey(ClothesColor, on_delete=models.PROTECT, related_name="dress_set", null=True,
                                    blank=True, db_column="원피스_색상")
    dress_sleeve_length = models.CharField(max_length=100, null=True, blank=True, db_column="원피스_소매기장")
    dress_material = models.CharField(max_length=100, null=True, blank=True, db_column="원피스_소재")
    dress_print = models.CharField(max_length=100, null=True, blank=True, db_column="원피스_프린트")
    dress_fit = models.CharField(max_length=100, null=True, blank=True, db_column="원피스_핏")
    dress_detail = models.CharField(max_length=255, null=True, blank=True, db_column="원피스_디테일")

    class Meta:
        db_table = "원피스"
        app_label = 'ui'


class UserInfo(models.Model):
    # Table: user_info
    # 사용자_식별자 integer [primary key]
    user_id = models.AutoField(primary_key=True, db_column="사용자_식별자")

    # [FK] 상의/하의/원피스 식별자 연결



    top_id = models.ForeignKey(TopBottom, on_delete=models.SET_NULL, related_name="user_tops", null=True, blank=True, db_column="상의_식별자")
    top_color = models.CharField(max_length=100, null=True, blank=True, db_column="상의_색상")
    top_category = models.CharField(max_length=100, null=True, blank=True, db_column="상의_카테고리")

    bottom_id = models.ForeignKey(TopBottom, on_delete=models.SET_NULL, related_name="user_bottoms", null=True, blank=True, db_column="하의_식별자")
    bottom_color = models.CharField(max_length=100, null=True, blank=True, db_column="하의_색상")
    bottom_category = models.CharField(max_length=100, null=True, blank=True, db_column="하의_카테고리")

    dress_id = models.ForeignKey(Dress, on_delete=models.SET_NULL, related_name="user_dresses", null=True, blank=True,db_column="원피스_식별자")
    dress_color = models.CharField(max_length=100, null=True, blank=True, db_column="원피스_색상")

    season = models.CharField(max_length=50, null=True, blank=True, db_column="계절")

    disliked_accord = models.CharField(max_length=50, null=True, blank=True, db_column="비선호_향조")

    class Meta:
        db_table = "user_info"
        app_label = 'ui'


# ==========================================
# 3. 향수 데이터 (Perfume)
# ==========================================

class Perfume(models.Model):
    # Table: perfume
    # perfume_id int [pk]
    perfume_id = models.IntegerField(primary_key=True,
                                     db_column="perfume_id")  # AutoField 대신 CSV ID 유지를 위해 IntegerField 추천

    url = models.CharField(max_length=500, null=True, blank=True, db_column="url")
    perfume_name = models.CharField(max_length=100, db_column="Perfume")
    brand = models.CharField(max_length=50, db_column="Brand")
    country = models.CharField(max_length=50, null=True, blank=True, db_column="Country")
    gender = models.CharField(max_length=20, null=True, blank=True, db_column="Gender")

    rating_value = models.FloatField(default=0.0, db_column="RatingValue")
    rating_count = models.IntegerField(default=0, db_column="RatingCount")
    year = models.IntegerField(null=True, blank=True, db_column="Year")

    top = models.TextField(null=True, blank=True, db_column="Top")
    middle = models.TextField(null=True, blank=True, db_column="Middle")
    base = models.TextField(null=True, blank=True, db_column="Base")

    # [FK] Main Accords -> perfume_color.mainaccord
    mainaccord1 = models.ForeignKey(PerfumeColor, on_delete=models.SET_NULL, related_name="p_accord1", null=True,
                                    blank=True, db_column="mainaccord1")
    mainaccord2 = models.ForeignKey(PerfumeColor, on_delete=models.SET_NULL, related_name="p_accord2", null=True,
                                    blank=True, db_column="mainaccord2")
    mainaccord3 = models.ForeignKey(PerfumeColor, on_delete=models.SET_NULL, related_name="p_accord3", null=True,
                                    blank=True, db_column="mainaccord3")
    mainaccord4 = models.ForeignKey(PerfumeColor, on_delete=models.SET_NULL, related_name="p_accord4", null=True,
                                    blank=True, db_column="mainaccord4")
    mainaccord5 = models.ForeignKey(PerfumeColor, on_delete=models.SET_NULL, related_name="p_accord5", null=True,
                                    blank=True, db_column="mainaccord5")

    def __str__(self):
        return f"{self.brand} - {self.perfume_name}"

    class Meta:
        db_table = "perfume"
        app_label = 'ui'


class PerfumeSeason(models.Model):
    # Table: perfume_season
    # perfume_id integer [primary key] -> OneToOne with Perfume
    perfume = models.OneToOneField(Perfume, primary_key=True, on_delete=models.CASCADE, related_name="season",
                                   db_column="perfume_id")
    spring = models.FloatField(default=0.0, db_column="spring")
    summer = models.FloatField(default=0.0, db_column="summer")
    fall = models.FloatField(default=0.0, db_column="fall")
    winter = models.FloatField(default=0.0, db_column="winter")

    class Meta:
        db_table = "perfume_season"
        app_label = 'ui'


class PerfumeClassification(models.Model):
    # Table: perfume_classification
    # perfume_id integer [primary key] -> OneToOne with Perfume
    perfume = models.OneToOneField(Perfume, primary_key=True, on_delete=models.CASCADE, related_name="classification",
                                   db_column="perfume_id")
    fragrance = models.CharField(max_length=255, null=True, blank=True, db_column="fragrance")

    class Meta:
        db_table = "perfume_classification"
        app_label = 'ui'


class Score(models.Model):
    # Table: score
    # perfume_id integer [primary key] -> OneToOne with Perfume
    perfume = models.OneToOneField(Perfume, primary_key=True, on_delete=models.CASCADE, related_name="score",
                                   db_column="perfume_id")
    season_score = models.FloatField(default=0.0, db_column="season_score")
    color_score = models.FloatField(default=0.0, db_column="color_score")
    style_score = models.FloatField(default=0.0, db_column="style_score")
    myscore = models.FloatField(default=0.0, db_column="myscore")

    class Meta:
        db_table = "score"
        app_label = 'ui'