### 2️⃣ 알고리즘 수식

$$Score = W_{color} \cdot \text{ColorScore} + W_{season} \cdot \text{SeasonScore} + W_{style} \cdot \text{StyleScore}$$

> **상세 수식 모드**
>
> $$Score = W_{color} \cdot \left[ 100 \times \left( 1 - \frac{\sqrt{(R_1 - R_2)^2 + (G_1 - G_2)^2 + (B_1 - B_2)^2}}{255\sqrt{3}} \right) \right] + W_{season} \cdot \left[ 100 \times \frac{V_{selected}}{V_{total}} \right] + W_{style} \cdot A_{ij}$$

### 3️⃣ 변수 상세 정의

| 변수명 | 설명 |
| :--- | :--- |
| $W_{color, season, style}$ | 각 지표별 반영 비중을 결정하는 가중치 (Weights) |
| $R, G, B$ | 의류 이미지 추출 색상(1)과 향수 대표 색상(2) 간의 RGB 값 |
| $V_{selected}$ | 타겟 계절에 대한 향수의 적합도 값 |
| $V_{total}$ | 해당 향수가 가진 전체 계절 적합도 값의 총합 |
| $A_{ij}$ | **스타일-향조 적합도 계수** |

> **$A_{ij}$ (Style Score) 핵심 요약**
> * **모델 기반 자동 분류:** 자체 구현 모델을 통해 의류 이미지의 스타일($i$) 자동 추출
> * **설문 기반 적합도 산출:** 추출된 스타일과 향수 향조($j$) 간의 조화도를 설문 데이터 기반 수치로 매핑
