from django.contrib import admin
from django.urls import path
from kgapp import views
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.contract_list, name='contract_list'),
    path('trends/', views.procurement_trends, name='procurement_trends'),
    path('contracts/', views.contract_list, name='contract_list'),
    path('sparql/', views.sparql_results, name='sparql_query'),
    path('institutions/', views.institution_list, name='institution_list'),
    path('institution/<int:institution_id>/', views.institution_detail, name='institution_detail'),
    path('suppliers/', views.supplier_list, name='supplier_list'),
    path('supplier/<int:supplier_id>/', views.supplier_detail, name='supplier_detail'),
    path('ontology/', views.ontology_view, name='ontology_view'),
    path('graph/', views.graph_view, name='graph_view'),
    path('knowledge_graph/', TemplateView.as_view(template_name='knowledge_graph.html'), name='knowledge_graph'),
    path('map/', views.map_view, name='map_view'),
    path('recommendations/', views.supplier_recommendations, name='supplier_recommendations'),
    path('semantic_search/', views.semantic_search, name='semantic_search'),
    path('supplier_network/', views.supplier_network_view, name='supplier_network'),
    path('contract_prediction/', views.contract_prediction_view, name='contract_prediction_view'),
    path('trend_analysis/', views.trend_analysis_view, name='trend_analysis'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
