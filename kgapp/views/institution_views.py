from django.core.paginator import Paginator
from django.shortcuts import get_object_or_404, render
from ..models import Contract, Institution
from django.db.models import F
import logging


logger = logging.getLogger(__name__)


def institution_list(request):
    # Get search query and sort order from GET parameters
    search_query = request.GET.get('search', '')
    sort_order = request.GET.get('sort', 'asc')  # Default to ascending order
    logger.info(f"Original search_query: {search_query}")

    # Get all institutions
    institutions = Institution.objects.all()

    # Apply search filter if query exists
    if search_query:
        institutions = institutions.filter(name__icontains=search_query)
        logger.info(f"Filtered institutions: {institutions.count()}")

    # Apply sorting
    if sort_order == 'desc':
        institutions = institutions.order_by(F('name').desc(nulls_last=True))
    else:
        institutions = institutions.order_by(F('name').asc(nulls_last=True))

    # Pagination
    paginator = Paginator(institutions, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Prepare context for template
    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'sort_order': sort_order,
    }
    return render(request, 'institution_list.html', context)


def institution_detail(request, institution_id):
    # Get institution by ID or return 404
    institution = get_object_or_404(Institution, id=institution_id)

    # Get all contracts for this institution
    contracts = Contract.objects.filter(institution=institution)

    # Prepare context for template
    context = {
        'institution': institution,
        'contracts': contracts,
    }
    return render(request, 'institution_details.html', context)
