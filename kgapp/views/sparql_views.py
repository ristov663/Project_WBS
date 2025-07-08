import rdflib
from django.shortcuts import render
from rdflib import Graph
from ..queries_script import get_all_queries
from .base import decode_uri


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
