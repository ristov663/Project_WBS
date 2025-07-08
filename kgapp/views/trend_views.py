from django.shortcuts import render
from ..models import Contract
from django.db.models import Sum
from django.db.models.functions import TruncYear, TruncMonth
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def procurement_trends(request):
    selected_year = request.GET.get('year')

    if selected_year and selected_year != 'all':
        # Aggregate data by month for the selected year
        monthly_data = Contract.objects.filter(date__year=selected_year).annotate(
            month=TruncMonth('date')).values('month').annotate(
            total_amount=Sum('amount')).order_by('month')

        labels = [data['month'].strftime('%B') for data in monthly_data]
        amounts = [float(data['total_amount']) for data in monthly_data]
        title = f"Procurement trend for {selected_year} year"
    else:
        # Aggregate data by year
        yearly_data = Contract.objects.annotate(year=TruncYear('date')).values('year').annotate(
            total_amount=Sum('amount')).order_by('year')

        labels = [data['year'].strftime('%Y') for data in yearly_data]
        amounts = [float(data['total_amount']) for data in yearly_data]
        title = "Procurement trend by year"

    # Data for the sum by year (for the sidebar list)
    contracts_by_year = Contract.objects.annotate(year=TruncYear('date')).values('year').annotate(
        total=Sum('amount')).order_by('-year')

    # List of all years for which we have data
    all_years = sorted(set(data['year'].year for data in contracts_by_year), reverse=True)

    context = {
        'labels': labels,
        'amounts': amounts,
        'contracts_by_year': contracts_by_year,
        'all_years': all_years,
        'selected_year': selected_year,
        'title': title,
    }
    return render(request, 'procurement_trends.html', context)


def trend_analysis_view(request):
    # Load predicted trends from CSV
    future_trends = pd.read_csv('kgapp/datasets/future_trends.csv')

    # Prepare data for display
    years = future_trends['Year'].tolist()
    amounts = future_trends['Predicted_Amount'].tolist()

    # Create a new graph for web display
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=future_trends, x='Year', y='Predicted_Amount', marker='o')
    plt.title('Predicted Trends in Public Procurement Amounts')
    plt.xlabel('Year')
    plt.ylabel('Predicted Total Amount')
    plt.tight_layout()
    plt.savefig("data/trend_analysis.png")
    plt.close()

    # Combine years and amounts for easy iteration in template
    data = list(zip(years, amounts))

    # Prepare context for template rendering
    context = {
        'years': years,
        'amounts': amounts,
        'data': data,
        'graph_image': '/data/trend_analysis.png',
    }
    return render(request, 'trend_analysis.html', context)
