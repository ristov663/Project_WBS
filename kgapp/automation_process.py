import csv
import os
import subprocess
import sys
import django
from urllib.parse import unquote
from django.utils.dateparse import parse_datetime
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import XSD
from flask import Flask, request, jsonify
from rdflib.plugins.sparql import prepareQuery

# Set up Django environment
sys.path.append(os.path.abspath('C:/Users/pc/Desktop/WbsProject'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'WbsProject.settings')

django.setup()

from kgapp.models import Institution, Supplier, Contract


# Function to run external Python scripts
def run_script(script_path):
    try:
        subprocess.run([sys.executable, script_path], check=True)
        print(f"Successfully executed script: {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing script {script_path}: {e}")
        sys.exit(1)


# Function to convert CSV data to RDF
def convert_to_rdf(csv_path):
    g = Graph()
    g.parse("kgapp/ontology/public_procurement.ttl", format="turtle")

    with open(csv_path, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Create RDF triples for each row in the CSV
            contract_uri = URIRef(f"http://example.org/contract/{row['Contract'].replace(' ', '_')}")
            g.add((contract_uri,
                   URIRef("http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/hasDescription"),
                   Literal(row['Contract'], datatype=XSD.string)))
            g.add((contract_uri, URIRef("http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/hasDate"),
                   Literal(row['Date'], datatype=XSD.dateTime)))
            if row['Amount'] != "0":
                g.add((contract_uri,
                       URIRef("http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/hasAmount"),
                       Literal(float(row['Amount']), datatype=XSD.float)))
            institution_uri = URIRef(f"http://example.org/institution/{row['Institution'].replace(' ', '_')}")
            g.add((contract_uri,
                   URIRef("http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/hasInstitution"),
                   institution_uri))
            supplier_uri = URIRef(f"http://example.org/supplier/{row['Supplier'].replace(' ', '_')}")
            g.add((contract_uri,
                   URIRef("http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/hasSupplier"),
                   supplier_uri))

    return g


# Function to update ontology (placeholder)
def update_ontology(g):
    # In this case, we don't perform additional ontology updates
    pass


# Function to validate data (placeholder)
def validate_data(g):
    # In this case, we don't perform additional validation
    pass


# Function to decode URI strings
def decode_uri(uri):
    return unquote(uri.split('/')[-1])


# Function to convert RDF data to Django models
def rdf_to_django_models(g):
    institution_map = {}
    supplier_map = {}

    # Create Institution and Supplier objects
    for subj, pred, obj in g:
        if pred == URIRef('http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/hasInstitution'):
            name = decode_uri(str(obj))
            institution, _ = Institution.objects.get_or_create(name=name)
            institution_map[obj] = institution
        elif pred == URIRef('http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/hasSupplier'):
            name = decode_uri(str(obj))
            supplier, _ = Supplier.objects.get_or_create(name=name)
            supplier_map[obj] = supplier

    # Create Contract objects
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


# Function to create SPARQL endpoint
def create_sparql_endpoint(g):
    app = Flask(__name__)

    @app.route('/')
    def home():
        return '''
        <h1>SPARQL Endpoint</h1>
        <p>Use /sparql endpoint with a 'query' parameter to execute SPARQL queries.</p>
        <p>Example: <a href="/sparql?query=SELECT * WHERE { ?s ?p ?o } LIMIT 10">/sparql?query=SELECT * WHERE { ?s ?p ?o } LIMIT 10</a></p>
        '''

    @app.route('/sparql', methods=['GET', 'POST'])
    def sparql_endpoint():
        query = request.args.get('query') or request.form.get('query')
        if not query:
            return jsonify({"error": "No query provided"}), 400

        try:
            prepared_query = prepareQuery(query)
            results = g.query(prepared_query)

            # Convert results to a list of dictionaries
            results_list = [
                {str(k): str(v) for k, v in row.items()}
                for row in results
            ]

            # Check if the client wants JSON or HTML
            if request.headers.get('Accept') == 'application/json':
                return jsonify(results_list)
            else:
                # Create an HTML table for the results
                table_html = '<table border="1"><tr>'
                if results_list:
                    table_html += ''.join(f'<th>{k}</th>' for k in results_list[0].keys())
                    table_html += '</tr>'
                    for row in results_list:
                        table_html += '<tr>' + ''.join(f'<td>{v}</td>' for v in row.values()) + '</tr>'
                table_html += '</table>'
                return f'<h2>Query Results</h2>{table_html}'

        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return app


# Main function to orchestrate the entire process
def main():
    # 1. Execute data scraping script
    run_script('kgapp/data_scraping.py')

    # 2. Convert to RDF
    g = convert_to_rdf('kgapp/datasets/all_contracts.csv')

    # 3. Validate data
    validate_data(g)

    # 4. Update ontology
    update_ontology(g)

    # 5. Save RDF
    g.serialize('kgapp/ontology/output.ttl', format='turtle')

    # 6. Convert RDF to Django models
    rdf_to_django_models(g)

    # 7. Create SPARQL endpoint
    app = create_sparql_endpoint(g)

    print("Automation completed successfully!")
    print("Starting SPARQL endpoint...")
    app.run(debug=True)


# Entry point of the script
if __name__ == "__main__":
    main()
