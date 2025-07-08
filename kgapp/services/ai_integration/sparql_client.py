from rdflib import Graph
from rdflib.term import URIRef, Literal
from urllib.parse import unquote


class SparqlClient:
    def __init__(self, ttl_file_path: str):
        self.graph = Graph()
        self.graph.parse(ttl_file_path, format="turtle")

    def query(self, sparql_query: str):
        results = self.graph.query(sparql_query)
        response = []
        for row in results:
            item = {}
            for k in row.labels:
                val = row[k]
                if isinstance(val, URIRef):
                    val_str = str(val)
                    if "%" in val_str:
                        val_str = unquote(val_str.split("/")[-1])
                    item[k] = val_str
                elif isinstance(val, Literal):
                    item[k] = val.value
                else:
                    item[k] = str(val)
            response.append(item)
        return response
