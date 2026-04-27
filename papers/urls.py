from django.urls import path
from . import views

urlpatterns = [
    # Public
    path('',                    views.index,          name='index'),
    path('api/search/',         views.search_api,     name='search_api'),

    # Auth
    path('login/',              views.login_view,     name='login'),
    path('logout/',             views.logout_view,    name='logout'),

    # Admin
    path('dashboard/',          views.dashboard,      name='dashboard'),
    path('api/upload/',         views.upload_file,    name='upload_file'),
    path('api/add-paper/',      views.add_paper,      name='add_paper'),
    path('api/delete/<int:paper_id>/', views.delete_paper, name='delete_paper'),
    path('api/papers/',         views.papers_list_api,name='papers_list'),
    path('api/stats/',          views.stats_api,      name='stats_api'),
]