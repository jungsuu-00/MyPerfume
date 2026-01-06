# 💐 MyPerfume  
**LG U+ Why Not SW Camp 8기**  
양원선 · 이정수 · 김태화


## 📌 프로젝트 개요
향수 지식이 없어도  
**지금의 나에게 어울리는 향을 이해하고 선택**할 수 있도록 설계한  
스타일·색감·계절 기반 개인화 향수 추천 서비스입니다.

- 의류 이미지 기반 스타일 분석
- 색감·계절 맥락을 반영한 추천 점수 계산
- 추천 이유를 자연어로 설명하는 구조


## 🎯 문제 정의 및 접근
기존 향수 추천 서비스는 향조·노트 중심 구조로  
비전문가에게 어렵고, 추천 근거가 명확하지 않다는 한계가 있습니다.

이를 해결하기 위해  
사람이 실제로 향수를 고를 때 고려하는 요소인  
**스타일 · 색감 · 계절**을 추천의 기준으로 설정했습니다.

## 📊 추천 로직
```
사용자 의류 이미지 & 선호 정보
        ↓
스타일 점수 / 색상 점수 / 계절 점수
        ↓
가중합(myscore) 계산
        ↓
Top-3 향수 추천 + 설명 생성
```
- 모든 점수는 정규화 후 가중치 튜닝
- Precision@k 기준 성능 개선 검증

## 🛠️ 설명 생성 (Explainability)
추천 결과에는 단순한 향수 리스트가 아닌,  
**스타일·색감·계절 맥락을 연결한 자연어 설명**을 함께 제공합니다.

- LLM은 설명 생성에만 사용  
- 추천 순위와 점수 계산은 데이터 기반 로직으로 분리 설계  

이를 통해 추천 결과에 대한 **이해도와 신뢰도**를 높였습니다.


## 📈 기대 효과
- 향수 지식이 없는 사용자도 추천 결과를 쉽게 이해 가능  
- 개인의 스타일과 상황에 맞는 향수 선택 경험 제공  
- 추천 근거가 명확한 설명 가능한 추천 시스템 구현  

## 📂 서비스 아키텍처
<br /> 

## 📂 프로토타입
<br />


## 💻 기술 스택

### Backend & Frontend
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white"> <img src="https://img.shields.io/badge/html5-E34F26?style=for-the-badge&logo=html5&logoColor=white"> <img src="https://img.shields.io/badge/css-1572B6?style=for-the-badge&logo=css3&logoColor=white"> <img src="https://img.shields.io/badge/Django_REST_Framework-ff1709?style=for-the-badge&logo=django&logoColor=white">

### Data / Modeling
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"> <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"> <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"> <img src="https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge&logo=lightgbm&logoColor=white"> <img src="https://img.shields.io/badge/CatBoost-FFCC00?style=for-the-badge&logo=catboost&logoColor=black">

### Infrastructure
<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"> <img src="https://img.shields.io/badge/Amazon EC2-FF9900?style=for-the-badge&logo=amazonec2&logoColor=white">
  <img src="https://img.shields.io/badge/Amazon S3-569A31?style=for-the-badge&logo=amazons3&logoColor=white"> <img src="https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white"> <img src="https://img.shields.io/badge/Nginx-009639?style=for-the-badge&logo=nginx&logoColor=white"> <img src="https://img.shields.io/badge/Gunicorn-499848?style=for-the-badge&logo=gunicorn&logoColor=white">

### Tools
<div>
  <img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">
  <img src="https://img.shields.io/badge/discord-5865F2?style=for-the-badge&logo=discord&logoColor=white">
  <img src="https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=Notion&logoColor=white" />
  <img src="https://img.shields.io/badge/PyCharm-000000?style=for-the-badge&logo=pycharm&logoColor=brightgreen" /> <img src="https://img.shields.io/badge/Visual Studio Code-007ACC.svg?style=for-the-badge&logo=Visual Studio Code&logoColor=white" />
  
</div>



## 💁‍♂️ 프로젝트 팀원

| | | |
|:---:|:---:|:---:|
| **[양원선](https://github.com/oneline47)** | **[이정수](https://github.com/jungsuu-00)** | **[김태화](https://github.com/kimtaehwa001)** |
| ![](https://github.com/oneline47.png?size=70) | ![](https://github.com/jungsuu-00.png?size=70) | ![](https://github.com/kimtaehwa001.png?size=70) |
| 팀장 · 기획<br/>데이터 분석 및 모델링 | 프론트엔드 & 백엔드 구축<br/>데이터 모델링 | 클라우드 엔지니어<br/>백엔드 구축 |