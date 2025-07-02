from rdflib import Graph, URIRef
import os
import sys
import django

from django.utils.dateparse import parse_datetime
from urllib.parse import unquote

# Set up Django environment
sys.path.append(os.path.abspath('C:/Users/pc/Desktop/WbsProject'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'WbsProject.settings')
django.setup()

from kgapp.models import Institution, Supplier, Contract

# Load RDF graph
g = Graph()
g.parse("kgapp/ontology/output.ttl", format="turtle")


# Helper function to decode URIs
def decode_uri(uri):
    return unquote(uri.split('/')[-1])

# Mapping institutions and suppliers
institution_map = {}
supplier_map = {}

for subj, pred, obj in g:
    if pred == URIRef('http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/hasInstitution'):
        name = decode_uri(str(obj))
        institution, _ = Institution.objects.get_or_create(name=name)
        institution_map[obj] = institution
    elif pred == URIRef('http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/hasSupplier'):
        name = decode_uri(str(obj))
        supplier, _ = Supplier.objects.get_or_create(name=name)
        supplier_map[obj] = supplier

# Processing contracts
for subj in g.subjects():
    if g.value(subj, URIRef('http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/hasAmount')):
        institution_uri = g.value(subj, URIRef(
            'http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/hasInstitution'))
        supplier_uri = g.value(subj, URIRef(
            'http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/hasSupplier'))

        institution = institution_map.get(institution_uri)
        supplier = supplier_map.get(supplier_uri)

        if not institution or not supplier:
            print(f"Error: Cannot find institution or supplier for contract {subj}")
            continue

        amount = float(
            g.value(subj, URIRef('http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/hasAmount')))
        date_str = str(
            g.value(subj, URIRef('http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/hasDate')))
        date = parse_datetime(date_str).date()
        description = decode_uri(str(g.value(subj, URIRef(
            'http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/hasDescription'))))

        Contract.objects.get_or_create(
            institution=institution,
            supplier=supplier,
            date=date,
            amount=amount,
            description=description
        )

print("Conversion completed.")
print(f"Number of contracts: {Contract.objects.count()}")
print(f"Number of institutions: {Institution.objects.count()}")
print(f"Number of suppliers: {Supplier.objects.count()}")

# New: Calculate and print relationships
total_links = Contract.objects.count()
unique_institution_supplier_pairs = Contract.objects.values('institution', 'supplier').distinct().count()

print(f"\nRelationship statistics:")
print(f"Total contractual links: {total_links}")
print(f"Unique institution-supplier pairs: {unique_institution_supplier_pairs}")
