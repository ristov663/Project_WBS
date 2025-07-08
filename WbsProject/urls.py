from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView

from kgapp.views.contract_views import contract_list, contract_prediction_view
from kgapp.views.institution_views import institution_list, institution_detail
from kgapp.views.supplier_views import supplier_list, supplier_detail, supplier_recommendations
from kgapp.views.graph_views import graph_view, map_view
from kgapp.views.kg_views import ontology_view
from kgapp.views.network_views import supplier_network_view
from kgapp.views.sparql_views import sparql_results
from kgapp.views.ai_views import semantic_search, chat_interface, conversational_chat
from kgapp.views.trend_views import procurement_trends, trend_analysis_view


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', contract_list, name='contract_list'),
    path('trends/', procurement_trends, name='procurement_trends'),
    path('contracts/', contract_list, name='contract_list'),
    path('sparql/', sparql_results, name='sparql_query'),
    path('institutions/', institution_list, name='institution_list'),
    path('institution/<int:institution_id>/', institution_detail, name='institution_detail'),
    path('suppliers/', supplier_list, name='supplier_list'),
    path('supplier/<int:supplier_id>/', supplier_detail, name='supplier_detail'),
    path('ontology/', ontology_view, name='ontology_view'),
    path('graph/', graph_view, name='graph_view'),
    path('knowledge_graph/', TemplateView.as_view(template_name='knowledge_graph.html'), name='knowledge_graph'),
    path('map/', map_view, name='map_view'),
    path('recommendations/', supplier_recommendations, name='supplier_recommendations'),
    path('semantic_search/', semantic_search, name='semantic_search'),
    path('supplier_network/', supplier_network_view, name='supplier_network'),
    path('contract_prediction/', contract_prediction_view, name='contract_prediction_view'),
    path('trend_analysis/', trend_analysis_view, name='trend_analysis'),
    path('conversational-chat/', conversational_chat, name='conversational_chat'),
    path('chat-interface/', chat_interface, name='chat_interface'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
