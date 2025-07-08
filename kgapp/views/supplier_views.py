from django.core.paginator import Paginator
from django.shortcuts import get_object_or_404, render
from ..models import Contract, Institution, Supplier
from django.db.models import Count, Sum, F, FloatField
from django.db.models.functions import Cast
import logging


logger = logging.getLogger(__name__)


def supplier_list(request):
    # Get search query and sort order from GET parameters
    search_query = request.GET.get('search', '')
    sort_order = request.GET.get('sort', 'asc')  # Default to ascending order
    logger.info(f"Original search_query: {search_query}")

    # Get all suppliers
    suppliers = Supplier.objects.all()

    # Apply search filter if query exists
    if search_query:
        suppliers = suppliers.filter(name__icontains=search_query)
        logger.info(f"Filtered suppliers: {suppliers.count()}")

    # Apply sorting
    if sort_order == 'desc':
        suppliers = suppliers.order_by(F('name').desc(nulls_last=True))
    else:
        suppliers = suppliers.order_by(F('name').asc(nulls_last=True))

    # Pagination
    paginator = Paginator(suppliers, 30)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Prepare context for template
    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'sort_order': sort_order,
    }
    return render(request, 'supplier_list.html', context)


def supplier_detail(request, supplier_id):
    # Get supplier by ID or return 404
    supplier = get_object_or_404(Supplier, id=supplier_id)

    # Get all contracts for this supplier
    contracts = Contract.objects.filter(supplier=supplier)

    # Prepare context for template
    context = {
        'supplier': supplier,
        'contracts': contracts,
    }
    return render(request, 'supplier_details.html', context)


def supplier_recommendations(request):
    institution_id = request.GET.get('institution')
    contract_type = request.GET.get('type')

    # Filter contracts based on institution and contract type
    contracts = Contract.objects.all()
    if institution_id:
        contracts = contracts.filter(institution_id=institution_id)
    if contract_type:
        contracts = contracts.filter(description__icontains=contract_type)

    # Aggregate data for suppliers
    supplier_data = (
        Supplier.objects.filter(contract__in=contracts)
        .annotate(
            contract_count=Count('contract'),
            total_amount=Sum('contract__amount'),
            weight=Cast(F('contract_count') * 0.7 + F('total_amount') * 0.3, FloatField())  # Cast to FloatField
        )
        .order_by('-weight')[:10]
    )

    # Prepare recommendations for the template
    recommendations = [
        {
            "name": supplier.name,
            "contract_count": supplier.contract_count,
            "total_amount": float(supplier.total_amount) if supplier.total_amount else 0,
            "weight": float(supplier.weight)
        }
        for supplier in supplier_data
    ]

    context = {
        "recommendations": recommendations,
        "institutions": Institution.objects.all(),
    }
    return render(request, 'supplier_recommendations.html', context)
