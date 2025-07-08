from django.shortcuts import render
from ..models import Contract, Supplier
import networkx as nx
import matplotlib.pyplot as plt


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
