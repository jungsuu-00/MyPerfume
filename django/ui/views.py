from django.shortcuts import render

# ==========================================
# 화면 렌더링 Views (HTML 페이지 연결)
# ==========================================

def home(request):
    return render(request, 'ui/home.html')

def for_me(request):
    return render(request, 'ui/for_me.html')

def for_someone(request):
    return render(request, 'ui/for_someone.html')

def result(request):
    return render(request, 'ui/result.html')

def result_someone(request):
    return render(request, 'ui/result_someone.html')

def my_note_style(request):
    request.session.pop("my_note_style", None)
    request.session.pop("my_note_perfumes", None)

    return render(request, "ui/my_note_style.html")

def my_note_perfume(request):
    return render(request, 'ui/my_note_perfume.html')

def my_note_result(request):
    return render(request, 'ui/my_note_result.html')

