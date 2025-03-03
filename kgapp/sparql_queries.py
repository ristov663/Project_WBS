import rdflib
from rdflib.plugins.sparql import prepareQuery
from urllib.parse import unquote
from rdflib import Graph


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


# 1. Листа на сите договори со нивните износи
query1 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?contract ?amount
WHERE {
    ?contract :hasAmount ?amount .
}
LIMIT 10
"""
execute_query(query1, "Листа на 10 договори со нивните износи")


# 2. Најскапите 5 договори
query2 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?contract ?amount
WHERE {
    ?contract :hasAmount ?amount .
}
ORDER BY DESC(?amount)
LIMIT 5
"""
execute_query(query2, "Топ 5 најскапи договори")


# 3. Вкупен износ на сите договори
query3 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT (SUM(?amount) AS ?total)
WHERE {
    ?contract :hasAmount ?amount .
}
"""
execute_query(query3, "Вкупен износ на сите договори")


# 4. Број на договори по година
query4 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?year (COUNT(?contract) AS ?count)
WHERE {
    ?contract :hasDate ?date .
    BIND(YEAR(?date) AS ?year)
}
GROUP BY ?year
ORDER BY ?year
"""
execute_query(query4, "Број на договори по година")


# 5. Договори со износ поголем од 1 милион
query5 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?contract ?amount
WHERE {
    ?contract :hasAmount ?amount .
    FILTER(?amount > 1000000)
}
LIMIT 10
"""
execute_query(query5, "Договори со износ поголем од 1 милион")


# 6. Институции со најголем број на договори
query6 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?institution (COUNT(?contract) AS ?contractCount)
WHERE {
    ?contract :hasInstitution ?institution .
}
GROUP BY ?institution
ORDER BY DESC(?contractCount)
LIMIT 5
"""
execute_query(query6, "Институции со најголем број на договори")


# 7. Добавувачи со највисока вкупна вредност на договори
query7 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?supplier (SUM(?amount) AS ?totalAmount)
WHERE {
    ?contract :hasSupplier ?supplier ;
              :hasAmount ?amount .
}
GROUP BY ?supplier
ORDER BY DESC(?totalAmount)
LIMIT 5
"""
execute_query(query7, "Добавувачи со највисока вкупна вредност на договори")


# 8. Просечна вредност на договорите по година
query8 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?year (AVG(?amount) AS ?avgAmount)
WHERE {
    ?contract :hasDate ?date ;
              :hasAmount ?amount .
    BIND(YEAR(?date) AS ?year)
}
GROUP BY ?year
ORDER BY ?year
"""
execute_query(query8, "Просечна вредност на договорите по година")


# 9. Договори со највисок износ за секоја институција
query9 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?institution (MAX(?amount) AS ?maxAmount)
WHERE {
    ?contract :hasInstitution ?institution ;
              :hasAmount ?amount .
}
GROUP BY ?institution
ORDER BY DESC(?maxAmount)
LIMIT 10
"""
execute_query(query9, "Највисок износ на договор по институција")


# 10. Број на договори по добавувач
query10 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?supplier (COUNT(?contract) AS ?contractCount)
WHERE {
    ?contract :hasSupplier ?supplier .
}
GROUP BY ?supplier
ORDER BY DESC(?contractCount)
LIMIT 10
"""
execute_query(query10, "Број на договори по добавувач")


# 11. Просечен износ на договори по месец и година
query11 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?year ?month (AVG(?amount) AS ?avgAmount)
WHERE {
    ?contract :hasDate ?date ;
              :hasAmount ?amount .
    BIND(YEAR(?date) AS ?year)
    BIND(MONTH(?date) AS ?month)
}
GROUP BY ?year ?month
ORDER BY ?year ?month
"""
execute_query(query11, "Просечен износ на договори по месец и година")


# 12. Договори со износ над просекот
query12 = """
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
"""
execute_query(query12, "Договори со износ над просекот")


# 13. Институции со најголем вкупен износ на договори
query13 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?institution (SUM(?amount) AS ?totalAmount)
WHERE {
    ?contract :hasInstitution ?institution ;
              :hasAmount ?amount .
}
GROUP BY ?institution
ORDER BY DESC(?totalAmount)
LIMIT 10
"""
execute_query(query13, "Институции со најголем вкупен износ на договори")


# 14. Број на договори по година и квартал
query14 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?year ?quarter (COUNT(?contract) AS ?contractCount)
WHERE {
    ?contract :hasDate ?date .
    BIND(YEAR(?date) AS ?year)
    BIND(CEIL(MONTH(?date) / 3) AS ?quarter)
}
GROUP BY ?year ?quarter
ORDER BY ?year ?quarter
"""
execute_query(query14, "Број на договори по година и квартал")


# 15. Договори со најкратки описи
query15 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?contract ?description
WHERE {
    ?contract :hasDescription ?description .
}
ORDER BY ASC(STRLEN(?description))
LIMIT 10
"""
execute_query(query15, "Договори со најкратки описи")


# 16. Институции со најголем просечен износ на договори
query16 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?institution (AVG(?amount) AS ?avgAmount)
WHERE {
    ?contract :hasInstitution ?institution ;
              :hasAmount ?amount .
}
GROUP BY ?institution
ORDER BY DESC(?avgAmount)
LIMIT 10
"""
execute_query(query16, "Институции со најголем просечен износ на договори")


# 17. Број на договори по месец за одредена година
query17 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?month (COUNT(?contract) AS ?count)
WHERE {
    ?contract :hasDate ?date .
    FILTER(YEAR(?date) = 2021)
    BIND(MONTH(?date) AS ?month)
}
GROUP BY ?month
ORDER BY ?month
"""
execute_query(query17, "Број на договори по месец за 2021 година")


# 18. Институции со најголем број на различни добавувачи
query18 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?institution (COUNT(DISTINCT ?supplier) AS ?supplierCount)
WHERE {
    ?contract :hasInstitution ?institution ;
              :hasSupplier ?supplier .
}
GROUP BY ?institution
ORDER BY DESC(?supplierCount)
LIMIT 10
"""
execute_query(query18, "Институции со најголем број на различни добавувачи")


# 19. Просечен износ на договори по квартал
query19 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?year ?quarter (AVG(?amount) AS ?avgAmount)
WHERE {
    ?contract :hasDate ?date ;
              :hasAmount ?amount .
    BIND(YEAR(?date) AS ?year)
    BIND(CEIL(MONTH(?date) / 3) AS ?quarter)
}
GROUP BY ?year ?quarter
ORDER BY ?year ?quarter
"""
execute_query(query19, "Просечен износ на договори по квартал")


# 20. Договори со најголема разлика помеѓу нивниот износ и просечниот износ
query20 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?contract ?amount (?amount - ?avgAmount AS ?difference)
WHERE {
    ?contract :hasAmount ?amount .
    {
        SELECT (AVG(?a) AS ?avgAmount)
        WHERE { ?c :hasAmount ?a }
    }
}
ORDER BY DESC(?difference)
LIMIT 10
"""
execute_query(query20, "Договори со најголема разлика од просечниот износ")


# 21. Институции со најголем пораст во вкупниот износ на договори од претходната година
query21 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?institution ?year ?totalAmount (?totalAmount - ?prevYearAmount AS ?increase)
WHERE {
    {
        SELECT ?institution ?year (SUM(?amount) AS ?totalAmount)
        WHERE {
            ?contract :hasInstitution ?institution ;
                      :hasDate ?date ;
                      :hasAmount ?amount .
            BIND(YEAR(?date) AS ?year)
        }
        GROUP BY ?institution ?year
    }
    {
        SELECT ?institution (?year - 1 AS ?prevYear) (SUM(?amount) AS ?prevYearAmount)
        WHERE {
            ?contract :hasInstitution ?institution ;
                      :hasDate ?date ;
                      :hasAmount ?amount .
            BIND(YEAR(?date) AS ?year)
        }
        GROUP BY ?institution ?year
    }
    FILTER(?year = ?prevYear + 1)
}
ORDER BY DESC(?increase)
LIMIT 10
"""
execute_query(query21, "Институции со најголем пораст во вкупниот износ на договори")


# 22. Договори со највисок износ за секоја година
query22 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?year (MAX(?amount) AS ?maxAmount)
WHERE {
    ?contract :hasAmount ?amount ;
              :hasDate ?date .
    BIND(YEAR(?date) AS ?year)
}
GROUP BY ?year
ORDER BY ?year
"""
execute_query(query22, "Договори со највисок износ по година")


# 23. Број на договори по институција
query23 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?institution (COUNT(?contract) AS ?contractCount)
WHERE {
    ?contract :hasInstitution ?institution .
}
GROUP BY ?institution
ORDER BY DESC(?contractCount)
LIMIT 10
"""
execute_query(query23, "Број на договори по институција")


# 24. Просечен износ на договори по институција
query24 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?institution (AVG(?amount) AS ?avgAmount)
WHERE {
    ?contract :hasInstitution ?institution ;
              :hasAmount ?amount .
}
GROUP BY ?institution
ORDER BY DESC(?avgAmount)
LIMIT 10
"""
execute_query(query24, "Просечен износ на договори по институција")


# 25. Договори со износ над 1 милијарда
query25 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?contract ?amount
WHERE {
    ?contract :hasAmount ?amount .
    FILTER(?amount > 1000000000)
}
ORDER BY DESC(?amount)
"""
execute_query(query25, "Договори со износ над 1 милијарда")


# 26. Број на договори по месец за сите години
query26 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?year ?month (COUNT(?contract) AS ?contractCount)
WHERE {
    ?contract :hasDate ?date .
    BIND(YEAR(?date) AS ?year)
    BIND(MONTH(?date) AS ?month)
}
GROUP BY ?year ?month
ORDER BY ?year ?month
"""
execute_query(query26, "Број на договори по месец за сите години")


# 27. Институции со најголем вкупен износ на договори за последната година
query27 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?institution (SUM(?amount) AS ?totalAmount)
WHERE {
    ?contract :hasInstitution ?institution ;
              :hasAmount ?amount ;
              :hasDate ?date .
    {
        SELECT (MAX(YEAR(?d)) AS ?maxYear)
        WHERE {
            ?c :hasDate ?d .
        }
    }
    FILTER(YEAR(?date) = ?maxYear)
}
GROUP BY ?institution
ORDER BY DESC(?totalAmount)
LIMIT 10
"""
execute_query(query27, "Институции со најголем вкупен износ за последната година")


# 28. Број на договори по година и квартал
query28 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?year ?quarter (COUNT(?contract) AS ?contractCount)
WHERE {
    ?contract :hasDate ?date .
    BIND(YEAR(?date) AS ?year)
    BIND(CEIL(MONTH(?date) / 3) AS ?quarter)
}
GROUP BY ?year ?quarter
ORDER BY ?year ?quarter
"""
execute_query(query28, "Број на договори по година и квартал")


# 29. Топ 10 добавувачи по вкупен износ на договори
query29 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?supplier (SUM(?amount) AS ?totalAmount)
WHERE {
    ?contract :hasSupplier ?supplier ;
              :hasAmount ?amount .
}
GROUP BY ?supplier
ORDER BY DESC(?totalAmount)
LIMIT 10
"""
execute_query(query29, "Топ 10 добавувачи по вкупен износ")


# 30. Договори склучени во последните 30 дена од последната година во податоците
query30 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?contract ?date ?amount
WHERE {
    ?contract :hasDate ?date ;
              :hasAmount ?amount .
    {
        SELECT (MAX(YEAR(?d)) AS ?maxYear)
        WHERE {
            ?c :hasDate ?d .
        }
    }
    FILTER(YEAR(?date) = ?maxYear && MONTH(?date) = 12 && DAY(?date) > 1)
}
ORDER BY DESC(?date)
"""
execute_query(query30, "Договори склучени во последните 30 дена од последната година")


# 31. Просечен број на договори по институција
query31 = """
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
execute_query(query31, "Просечен број на договори по институција")


# 32. Институции со најголем пораст во бројот на договори од претходната година
query32 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?institution ?year ?contractCount (?contractCount - ?prevYearCount AS ?increase)
WHERE {
    {
        SELECT ?institution ?year (COUNT(?contract) AS ?contractCount)
        WHERE {
            ?contract :hasInstitution ?institution ;
                      :hasDate ?date .
            BIND(YEAR(?date) AS ?year)
        }
        GROUP BY ?institution ?year
    }
    {
        SELECT ?institution (?year - 1 AS ?prevYear) (COUNT(?contract) AS ?prevYearCount)
        WHERE {
            ?contract :hasInstitution ?institution ;
                      :hasDate ?date .
            BIND(YEAR(?date) AS ?year)
        }
        GROUP BY ?institution ?year
    }
    FILTER(?year = ?prevYear + 1)
}
ORDER BY DESC(?increase)
LIMIT 10
"""
execute_query(query32, "Институции со најголем пораст во бројот на договори")


# 33. Договори со износ значително над просекот за таа институција
query33 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?institution ?contract ?amount ?avgAmount
WHERE {
    ?contract :hasInstitution ?institution ;
              :hasAmount ?amount .
    {
        SELECT ?institution (AVG(?a) AS ?avgAmount)
        WHERE {
            ?c :hasInstitution ?institution ;
               :hasAmount ?a .
        }
        GROUP BY ?institution
    }
    FILTER(?amount > ?avgAmount * 2)
}
ORDER BY DESC(?amount)
LIMIT 10
"""
execute_query(query33, "Договори значително над просекот за институцијата")


# 34. Најчести комбинации на институција и добавувач
query34 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?institution ?supplier (COUNT(?contract) AS ?contractCount)
WHERE {
    ?contract :hasInstitution ?institution ;
              :hasSupplier ?supplier .
}
GROUP BY ?institution ?supplier
ORDER BY DESC(?contractCount)
LIMIT 10
"""
execute_query(query34, "Најчести комбинации на институција и добавувач")


# 35. Институции со најголем број на уникатни добавувачи
query35 = """
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?institution (COUNT(DISTINCT ?supplier) AS ?uniqueSuppliers)
WHERE {
    ?contract :hasInstitution ?institution ;
              :hasSupplier ?supplier .
}
GROUP BY ?institution
ORDER BY DESC(?uniqueSuppliers)
LIMIT 10
"""
execute_query(query35, "Институции со најголем број на уникатни добавувачи")
