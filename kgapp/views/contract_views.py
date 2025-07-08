from django.core.paginator import Paginator
from django.shortcuts import render
from ..models import Contract
from django.db.models import Sum, Q
from django.db.models.functions import TruncYear
from ..contract_amount_prediction import predict_contract_amount, get_unique_values, visualize_historical_trends, \
    analyze_factors
import logging


logger = logging.getLogger(__name__)


def contract_list(request):
    contracts = Contract.objects.all()

    # Search functionality
    search_query = request.GET.get('search', '')
    logger.info(f"Original search_query: {search_query}")

    if search_query:
        contracts = contracts.filter(
            Q(institution__name__icontains=search_query) |
            Q(supplier__name__icontains=search_query)
        )
        logger.info(f"Filtered contracts: {contracts.count()}")

    # Filter by year
    year = request.GET.get('year')
    if year and year != 'None':
        contracts = contracts.filter(date__year=year)

    # Filter by date and price
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    min_amount = request.GET.get('min_amount')
    max_amount = request.GET.get('max_amount')

    if start_date and start_date != 'None':
        contracts = contracts.filter(date__gte=start_date)
    if end_date and end_date != 'None':
        contracts = contracts.filter(date__lte=end_date)
    if min_amount and min_amount != 'None':
        contracts = contracts.filter(amount__gte=float(min_amount))
    if max_amount and max_amount != 'None':
        contracts = contracts.filter(amount__lte=float(max_amount))

    # Sorting based on the `sort` parameter
    sort = request.GET.get('sort', '-amount')
    if sort in ['amount', '-amount', 'institution', '-institution']:
        contracts = contracts.order_by(sort)

    # Pagination
    paginator = Paginator(contracts, 40)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Annual sum of contracts
    contracts_by_year = Contract.objects.annotate(year=TruncYear('date')).values('year').annotate(
        total=Sum('amount')).order_by('year')

    # List of all years for which we have contracts
    all_years = Contract.objects.dates('date', 'year', order='DESC')

    context = {
        'page_obj': page_obj,
        'start_date': start_date,
        'end_date': end_date,
        'min_amount': min_amount,
        'max_amount': max_amount,
        'contracts_by_year': contracts_by_year,
        'sort': sort,
        'all_years': all_years,
        'selected_year': year,
        'search_query': search_query,
    }
    return render(request, 'contract_list.html', context)


def contract_prediction_view(request):
    # Get unique institutions and suppliers
    institutions, suppliers = get_unique_values()

    if request.method == 'POST':
        # Extract form data
        institution = request.POST.get('institution')
        contract_description = request.POST.get('contract_description')
        supplier = request.POST.get('supplier')

        # Predict contract amount
        predicted_amount = predict_contract_amount(institution, contract_description, supplier)

        # Generate visualizations
        visualize_historical_trends(institution)
        analyze_factors()

        # Prepare context for results template
        context = {
            'predicted_amount': predicted_amount,
            'institution': institution,
            'contract_description': contract_description,
            'supplier': supplier,
            'institutions': institutions,
            'suppliers': suppliers,
            'show_results': True,
            'historical_trend_image': '../../data/historical_trend.png',
            'factor_importance_image': '../../data/factor_importance.png'
        }
        return render(request, 'contract_prediction_result.html', context)

    # Prepare context for form template
    context = {
        'institutions': institutions,
        'suppliers': suppliers,
        'show_results': False
    }
    return render(request, 'contract_prediction_form.html', context)
