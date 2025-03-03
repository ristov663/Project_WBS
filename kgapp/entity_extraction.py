import spacy
import pandas as pd
import networkx as nx
from pyvis.network import Network
from django.shortcuts import render
from django.conf import settings
import os

# Load the Macedonian language model
nlp = spacy.load("mk_core_news_sm")


# Function to extract named entities from text
def extract_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })
    return entities


# Function to process the dataset and create a graph
def process_dataset(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')

    all_entities = []
    graph = nx.Graph()

    for _, row in df.iterrows():
        institution = row['Institution']
        contract = row['Contract']
        supplier = row['Supplier']
        date = row['Date']
        amount = row['Amount']

        # Add nodes to the graph
        graph.add_node(institution, title=institution, group='Institution')
        graph.add_node(supplier, title=supplier, group='Supplier')

        # Add edges to the graph
        graph.add_edge(institution, supplier, title=f"{contract}\nDate: {date}\nAmount: {amount}")

        # Extract entities from contract description
        contract_entities = extract_entities(contract)
        all_entities.extend(contract_entities)

        # Add extracted entities to the graph
        for entity in contract_entities:
            graph.add_node(entity['text'], title=entity['text'], group=entity['label'])
            graph.add_edge(institution, entity['text'], title='has_entity')
            graph.add_edge(supplier, entity['text'], title='has_entity')

    return graph, all_entities


# Function to create an interactive graph
def create_interactive_graph(graph):
    net = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(graph)
    net.toggle_physics(True)
    net.show_buttons(filter_=['physics'])

    # Save the HTML file
    static_path = os.path.join(settings.STATIC_ROOT, 'knowledge_graph.html')
    net.save_graph(static_path)


# Django view function to render the knowledge graph
def knowledge_graph_view(request):
    file_path = 'kgapp/datasets/datasets_converted.csv'
    graph, entities = process_dataset(file_path)
    create_interactive_graph(graph)

    context = {
        'entities': entities[:100]  # Display first 100 entities
    }
    return render(request, 'knowledge_graph.html', context)
