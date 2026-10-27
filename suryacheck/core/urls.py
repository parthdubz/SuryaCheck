from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("run-batch/", views.run_batch, name="run_batch"),
    path("fleet/", views.fleet_dashboard, name="fleet_dashboard"),
    path("inverter/<str:inverter_id>/", views.inverter_detail, name="inverter_detail"),
    path("alert/<int:alert_id>/dismiss/", views.dismiss_alert, name="dismiss_alert"),
    path("break-it-live/", views.break_it_live, name="break_it_live"),
    path("results/", views.results_view, name="results_view"),
    path("limits/", views.limits_view, name="limits_view"),
]
