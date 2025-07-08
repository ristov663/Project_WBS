from django.shortcuts import render
from ..models import Contract, Institution, Supplier


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
