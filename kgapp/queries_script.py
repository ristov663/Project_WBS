import rdflib
from urllib.parse import unquote
from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery


g = Graph()
g.parse("kgapp/ontology/output.ttl", format="turtle")


def decode_uri(uri):
    return unquote(uri.split('/')[-1])


def format_literal(literal):
    if isinstance(literal, rdflib.Literal):
        return f"{literal.value} ({literal.datatype})"
    return str(literal)


def execute_query(query_string, query_name):
    print(f"\n{query_name}:")
    query = prepareQuery(query_string, initNs={
        "": "http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/"
    })
    results = g.query(query)
    for row in results:
        decoded_row = [decode_uri(str(item)) if isinstance(item, rdflib.term.URIRef) else format_literal(item) for item in row]
        print(decoded_row)


QUERIES = {

    "Топ 5 најскапи договори": """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?contract ?amount
WHERE {
    ?contract :hasAmount ?amount .
}
ORDER BY DESC(?amount)
LIMIT 5
""",

    "Вкупен износ на сите договори": """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT (SUM(?amount) AS ?total)
WHERE {
    ?contract :hasAmount ?amount .
}
""",

    "Број на договори по година": """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?year (COUNT(?contract) AS ?count)
WHERE {
    ?contract :hasDate ?date .
    BIND(YEAR(?date) AS ?year)
}
GROUP BY ?year
ORDER BY ?year
""",

    "Договори со износ поголем од 1 милион евра": """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?contract ?amount
WHERE {
    ?contract :hasAmount ?amount .
    FILTER(?amount > 61500000)
}
LIMIT 10
""",

    "Институции со најголем број на договори": """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?institution (COUNT(?contract) AS ?contractCount)
WHERE {
    ?contract :hasInstitution ?institution .
}
GROUP BY ?institution
ORDER BY DESC(?contractCount)
LIMIT 5
""",

    "Добавувачи со највисока вкупна вредност на договори": """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?supplier (SUM(?amount) AS ?totalAmount)
WHERE {
    ?contract :hasSupplier ?supplier ;
              :hasAmount ?amount .
}
GROUP BY ?supplier
ORDER BY DESC(?totalAmount)
LIMIT 5
""",

    "Просечна вредност на договорите по година": """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?year (AVG(?amount) AS ?avgAmount)
WHERE {
    ?contract :hasDate ?date ;
              :hasAmount ?amount .
    BIND(YEAR(?date) AS ?year)
}
GROUP BY ?year
ORDER BY ?year
""",

    "Највисок износ на договор по институција": """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?institution (MAX(?amount) AS ?maxAmount)
WHERE {
    ?contract :hasInstitution ?institution ;
              :hasAmount ?amount .
}
GROUP BY ?institution
ORDER BY DESC(?maxAmount)
LIMIT 10
""",

    "Број на договори по добавувач": """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?supplier (COUNT(?contract) AS ?contractCount)
WHERE {
    ?contract :hasSupplier ?supplier .
}
GROUP BY ?supplier
ORDER BY DESC(?contractCount)
LIMIT 10
""",

    "Договори со износ над просекот": """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?contract ?amount
WHERE {
    ?contract :hasAmount ?amount .
    {
        SELECT (AVG(?a) AS ?avgAmount)
        WHERE {
            ?c :hasAmount ?a .
        }
    }
    FILTER(?amount > ?avgAmount)
}
ORDER BY DESC(?amount)
LIMIT 10
""",

    "Институции со најголем вкупен износ на договори": """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?institution (SUM(?amount) AS ?totalAmount)
WHERE {
    ?contract :hasInstitution ?institution ;
              :hasAmount ?amount .
}
GROUP BY ?institution
ORDER BY DESC(?totalAmount)
LIMIT 10
""",

    "Институции со најголем просечен износ на договори": """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?institution (AVG(?amount) AS ?avgAmount)
WHERE {
    ?contract :hasInstitution ?institution ;
              :hasAmount ?amount .
}
GROUP BY ?institution
ORDER BY DESC(?avgAmount)
LIMIT 10
""",

    "Институции со најголем број на различни добавувачи": """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?institution (COUNT(DISTINCT ?supplier) AS ?supplierCount)
WHERE {
    ?contract :hasInstitution ?institution ;
              :hasSupplier ?supplier .
}
GROUP BY ?institution
ORDER BY DESC(?supplierCount)
LIMIT 10
""",

    "Договори со највисок износ по година": """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?year (MAX(?amount) AS ?maxAmount)
WHERE {
    ?contract :hasAmount ?amount ;
              :hasDate ?date .
    BIND(YEAR(?date) AS ?year)
}
GROUP BY ?year
ORDER BY ?year
""",

    "Топ 10 добавувачи по вкупен износ": """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?supplier (SUM(?amount) AS ?totalAmount)
WHERE {
    ?contract :hasSupplier ?supplier ;
              :hasAmount ?amount .
}
GROUP BY ?supplier
ORDER BY DESC(?totalAmount)
LIMIT 10
""",

    "Просечен број на договори по институција": """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT (AVG(?count) AS ?avgContracts)
WHERE {
    SELECT ?institution (COUNT(?contract) AS ?count)
    WHERE {
        ?contract :hasInstitution ?institution .
    }
    GROUP BY ?institution
}
"""
}


def get_all_queries():
    return QUERIES
