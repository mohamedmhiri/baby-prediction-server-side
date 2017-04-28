from django.conf.urls import url
from api import views

urlpatterns = [
    url(r'^data/$', views.data_list),
    url(r'^data/(?P<pk>[0-9]+)/$', views.data_detail),
    url(r'^forest/$', views.random_forest),
    url(r'^load/$', views.data_generator)
]