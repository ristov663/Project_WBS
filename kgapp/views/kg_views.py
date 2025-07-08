from django.shortcuts import render
from rdflib import Graph


def ontology_view(request):
    # Load the ontology
    g = Graph()
    g.parse("kgapp/ontology/public_procurement.ttl", format="turtle")

    # Extract classes, object properties, and data properties
    classes = []
    object_properties = []
    data_properties = []

    for s, p, o in g:
        # Identify classes
        if str(p) == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" and str(o) \
                == "http://www.w3.org/2002/07/owl#Class":
            classes.append(str(s))
        # Identify object properties
        elif str(p) == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" and str(o) \
                == "http://www.w3.org/2002/07/owl#ObjectProperty":
            object_properties.append(str(s))
        # Identify data properties
        elif str(p) == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" and str(o) \
                == "http://www.w3.org/2002/07/owl#DatatypeProperty":
            data_properties.append(str(s))

    # Prepare context for the template
    context = {
        'classes': classes,
        'object_properties': object_properties,
        'data_properties': data_properties,
    }
    # Render the template with the context
    return render(request, 'ontology_view.html', context)
