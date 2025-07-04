import rdflib
import cohere
from django.core.paginator import Paginator
from django.shortcuts import render, get_object_or_404
from .models import Contract, Institution, Supplier
from rdflib import Graph, URIRef
from .queries_script import get_all_queries
from urllib.parse import unquote
from django.db.models import Count, Sum, F, FloatField, Q
from django.db.models.functions import TruncYear, TruncMonth
from django.db.models.functions import Cast
import networkx as nx
import matplotlib.pyplot as plt
from django.views.generic import TemplateView
from .contract_amount_prediction import predict_contract_amount, get_unique_values, visualize_historical_trends, \
    analyze_factors
import pandas as pd
import seaborn as sns
import logging
import google.generativeai as genai
import os


class KnowledgeGraphView(TemplateView):
    template_name = 'knowledge_graph.html'


def ontology_view(request):
    # Load the ontology
    g = Graph()
    g.parse("kgapp/ontology/public_procurement.ttl", format="turtle")

    # Extract classes, object properties, and data properties
    classes = []
    object_properties = []
    data_properties = []

    for s, p, o in g:
        # Identify classes
        if str(p) == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" and str(o) \
                == "http://www.w3.org/2002/07/owl#Class":
            classes.append(str(s))
        # Identify object properties
        elif str(p) == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" and str(o) \
                == "http://www.w3.org/2002/07/owl#ObjectProperty":
            object_properties.append(str(s))
        # Identify data properties
        elif str(p) == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" and str(o) \
                == "http://www.w3.org/2002/07/owl#DatatypeProperty":
            data_properties.append(str(s))

    # Prepare context for the template
    context = {
        'classes': classes,
        'object_properties': object_properties,
        'data_properties': data_properties,
    }
    # Render the template with the context
    return render(request, 'ontology_view.html', context)


def decode_uri(uri):
    return unquote(uri.split('/')[-1])


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


def sparql_results(request):
    # Load the RDF graph
    g = Graph()
    g.parse("kgapp/ontology/output.ttl", format="turtle")

    # Get all predefined SPARQL queries
    all_queries = get_all_queries()
    results = {}

    # Execute each query and process the results
    for query_name, query in all_queries.items():
        query_results = g.query(query)
        decoded_results = []
        for row in query_results:
            # Decode URI references in the results
            decoded_row = [decode_uri(str(item)) if isinstance(item, rdflib.term.URIRef) else str(item) for item in row]
            decoded_results.append(decoded_row)
        results[query_name] = decoded_results

    # Prepare context for template
    context = {'results': results}
    return render(request, 'sparql_results.html', context)


def graph_view(request):
    nodes = []
    edges = []

    # Add institutions as nodes
    institutions = Institution.objects.all()
    for institution in institutions:
        nodes.append({"data": {"id": f"inst_{institution.id}", "label": institution.name}})

    # Add suppliers as nodes
    suppliers = Supplier.objects.all()
    for supplier in suppliers:
        nodes.append({"data": {"id": f"supp_{supplier.id}", "label": supplier.name}})

    # Add contracts as edges (relationships)
    contracts = Contract.objects.all()
    for contract in contracts:
        edges.append({
            "data": {
                "source": f"inst_{contract.institution.id}",
                "target": f"supp_{contract.supplier.id}",
                "label": f"{contract.amount} MKD"
            }
        })

    # Prepare context for template
    context = {
        "nodes": nodes,
        "edges": edges,
    }
    return render(request, 'graph_view.html', context)


def map_view(request):
    institutions = Institution.objects.all()
    suppliers = Supplier.objects.all()

    # Prepare data for map visualization
    context = {
        'institutions': [
            {"name": inst.name, "latitude": inst.latitude, "longitude": inst.longitude}
            for inst in institutions if inst.latitude and inst.longitude
        ],
        'suppliers': [
            {"name": supp.name, "latitude": supp.latitude, "longitude": supp.longitude}
            for supp in suppliers if supp.latitude and supp.longitude
        ],
    }
    return render(request, 'map_view.html', context)


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


genai.configure(api_key="API_KEY")
co = cohere.Client("API_KEY")


def generate_sparql_from_natural_language(query):
    # Initialize the Gemini Pro model
    model = genai.GenerativeModel("gemini-pro")

    # Construct the prompt for SPARQL query generation
    prompt = f"""
    You are a SPARQL query generator for a public procurement ontology.
    Convert the following natural language question into a valid SPARQL query.

    Question: {query}
    SPARQL Query (without markdown or explanations):
    """

    # Generate the SPARQL query using the AI model
    response = model.generate_content(prompt)

    sparql_query = response.text.strip()

    # Remove potential Markdown tags
    sparql_query = sparql_query.replace("``````", "").strip()

    print("Generated SPARQL Query:\n", sparql_query)  # Logging for verification

    return sparql_query


def semantic_search(request):
    query = request.GET.get('query', '')

    if not query:
        return render(request, 'semantic_search.html', {"results": []})

    try:
        # Generate SPARQL query from natural language
        sparql_query = generate_sparql_from_natural_language(query)

        # Fix prefixes and property names
        sparql_query = sparql_query.replace("public_procurement:", ":")
        sparql_query = sparql_query.replace("procurement:", ":")
        sparql_query = sparql_query.replace("pp:", ":")
        sparql_query = sparql_query.replace("hasValue", "hasAmount")

        # Load the RDF graph
        g = Graph()
        g.parse("kgapp/ontology/output.ttl", format="turtle")

        # Add namespace
        g.bind("", URIRef("http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/"))

        # Execute the SPARQL query
        results = g.query(sparql_query)

        # Process and format the results
        response = []
        for row in results:
            response.append({
                "institution": getattr(row, 'institution', 'N/A'),
            })

        return render(request, 'semantic_search.html', {"results": response, "query": query})

    except Exception as e:
        return render(request, 'semantic_search.html', {"error": str(e), "query": query})


def analyze_supplier_network():
    # Fetch contract data
    contracts = Contract.objects.all().values('institution', 'supplier')

    # Create a graph
    g = nx.Graph()

    # Add edges to the graph (weight is the number of contracts)
    for contract in contracts:
        if g.has_edge(contract['supplier'], contract['institution']):
            g[contract['supplier']][contract['institution']]['weight'] += 1
        else:
            g.add_edge(contract['supplier'], contract['institution'], weight=1)

    # Calculate centrality
    degree_centrality = nx.degree_centrality(g)

    # Identify top 10 suppliers by centrality
    top_suppliers = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    top_supplier_ids = [supplier[0] for supplier in top_suppliers]

    # Create subgraph with top 10 suppliers
    subgraph = g.subgraph(top_supplier_ids).copy()

    return subgraph, degree_centrality, top_suppliers


def visualize_network(g, degree_centrality):
    plt.figure(figsize=(12, 8))

    # Position the nodes
    pos = nx.spring_layout(g, seed=42)

    # Draw the nodes
    nx.draw_networkx_nodes(
        g,
        pos,
        node_size=[degree_centrality[node] * 1000 for node in g.nodes()],
        node_color='skyblue',
        alpha=0.9
    )

    # Draw the edges
    nx.draw_networkx_edges(
        g,
        pos,
        width=1.5,
        alpha=0.7,
        edge_color='gray'
    )

    # Add labels with supplier names
    nx.draw_networkx_labels(
        g,
        pos,
        labels={node: node for node in g.nodes()},
        font_size=10,
        font_color='black'
    )

    plt.title("Мрежа на соработка помеѓу топ 10 добавувачи")
    plt.axis('off')
    plt.tight_layout()

    # Save the image
    plt.savefig('data/top10_supplier_network.png')
    plt.close()


def supplier_network_view(request):
    # Analyze the supplier network
    g, degree_centrality, top_suppliers = analyze_supplier_network()

    # Visualize the subgraph with top 10 suppliers
    visualize_network(g, degree_centrality)

    # Get names for top 10 suppliers
    supplier_objects = Supplier.objects.filter(id__in=[s[0] for s in top_suppliers])
    supplier_dict = {s.id: s for s in supplier_objects}

    # Prepare detailed information for top suppliers
    top_suppliers_with_details = [
        {
            'id': s[0],
            'name': supplier_dict.get(s[0]).name if s[0] in supplier_dict else f"Supplier {s[0]}",
            'centrality': s[1]
        } for s in top_suppliers
    ]

    # Prepare context for template rendering
    context = {
        'top_suppliers': top_suppliers_with_details,
        'network_image': '../../data/top10_supplier_network.png'
    }

    # Render the template with the context
    return render(request, 'supplier_network.html', context)


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


import os
from dotenv import load_dotenv
import re
import google.generativeai as genai

# Load environment variables
load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY is not set")
genai.configure(api_key=api_key)


def extract_sparql_from_code_block(text: str) -> str:
    """
    If Gemini wraps the SPARQL query in ```code blocks```, strip them.
    """
    match = re.search(r"```(?:sparql)?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def generate_sparql_from_natural_language(user_question: str) -> str:
    """
    Calls Gemini to generate a SPARQL query based on your ontology and a natural language question.
    Uses the correct pattern: contracts are identified by their properties, not by rdf:type.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
You are an expert SPARQL query generator for a public procurement RDF ontology.

Always use this prefix:
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>

Classes:
- :Contract (but NOT declared with rdf:type)
- :Institution
- :Supplier

Properties:
- :hasAmount (float)
- :hasDate (date)
- :hasInstitution (Institution)
- :hasSupplier (Supplier)
- :hasDescription (string)

✅ IMPORTANT: Contracts do NOT have an explicit rdf:type. Never use ?contract a :Contract .
✅ Always identify ?contract by matching on its properties (e.g., ?contract :hasAmount ?amount).
✅ For dates, always extract years using: BIND(YEAR(?date) AS ?year)
✅ Use GROUP BY, ORDER BY and LIMIT when needed.
✅ Use clear variable names.
✅ Return ONLY valid SPARQL — no markdown or code fences.

Examples of GOOD patterns:

-- Top 5 contracts by amount:
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?contract ?amount
WHERE {{
  ?contract :hasAmount ?amount .
}}
ORDER BY DESC(?amount)
LIMIT 5

-- Number of contracts per year:
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?year (COUNT(?contract) AS ?contractCount)
WHERE {{
  ?contract :hasDate ?date .
  BIND(YEAR(?date) AS ?year)
}}
GROUP BY ?year
ORDER BY ?year

-- Top 5 institutions by total contract amount:
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?institution (SUM(?amount) AS ?totalAmount)
WHERE {{
  ?contract :hasInstitution ?institution ;
            :hasAmount ?amount .
}}
GROUP BY ?institution
ORDER BY DESC(?totalAmount)
LIMIT 5

---
Now convert this question to SPARQL:
{user_question}
    """.strip()

    # Call Gemini
    response = model.generate_content([{"role": "user", "parts": [prompt]}])

    # Extract and clean the query
    raw_sparql = response.text.strip()
    sparql_query = extract_sparql_from_code_block(raw_sparql)

    print("\n[Gemini] Generated SPARQL:\n", sparql_query)

    return sparql_query


from urllib.parse import unquote
from rdflib.term import URIRef, Literal


def semantic_search(request):
    query = request.GET.get('query', '')

    if not query:
        return render(request, 'semantic_search.html', {"results": []})

    try:
        sparql_query = generate_sparql_from_natural_language(query)

        g = Graph()
        g.parse("kgapp/ontology/output.ttl", format="turtle")

        results = g.query(sparql_query)

        response = []
        for row in results:
            item = {}
            for k in row.labels:
                val = row[k]
                if isinstance(val, URIRef):
                    val_str = str(val)
                    if "%" in val_str:
                        val_str = unquote(val_str.split("/")[-1])
                    else:
                        val_str = val_str
                    item[k] = val_str
                elif isinstance(val, Literal):
                    item[k] = val.value  # will be int for year/count!
                else:
                    item[k] = str(val)
            response.append(item)

        return render(request, 'semantic_search.html', {"results": response, "query": query})

    except Exception as e:
        return render(request, 'semantic_search.html', {"error": str(e), "query": query})
