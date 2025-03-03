import csv
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import XSD

# Create RDF graph
g = Graph()
g.parse("kgapp/ontology/public_procurement.ttl", format="turtle")

# Path to CSV file
csv_file = "kgapp/datasets/datasets_converted.csv"


# Read data from CSV and add to the graph
with open(csv_file, newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)

    for row in reader:
        # Create a new URI for the contract
        contract_uri = URIRef(f"http://example.org/contract/{row['Contract'].replace(' ', '_')}")

        # Add contract to the graph
        g.add((contract_uri,
               URIRef("http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/hasDescription"),
               Literal(row['Contract'], datatype=XSD.string)))
        g.add((contract_uri, URIRef("http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/hasDate"),
               Literal(row['Date'], datatype=XSD.dateTime)))

        # Add amount
        if row['Amount'] != "0":
            g.add((contract_uri,
                   URIRef("http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/hasAmount"),
                   Literal(float(row['Amount']), datatype=XSD.float)))

        # Add Institution
        institution_uri = URIRef(f"http://example.org/institution/{row['Institution'].replace(' ', '_')}")
        g.add((contract_uri,
               URIRef("http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/hasInstitution"),
               institution_uri))

        # Add Supplier
        supplier_uri = URIRef(f"http://example.org/supplier/{row['Supplier'].replace(' ', '_')}")
        g.add((contract_uri, URIRef("http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/hasSupplier"),
               supplier_uri))

# Write the result in Turtle format
g.serialize("kgapp/ontology/output.ttl", format="turtle")
