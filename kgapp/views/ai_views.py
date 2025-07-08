from django.shortcuts import render
from kgapp.services.ai_integration.sparql_generator import ask_ai
from kgapp.services.ai_integration.sparql_client import SparqlClient
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rdflib import Graph, URIRef, Literal
from urllib.parse import unquote

TTL_FILE_PATH = "kgapp/ontology/output.ttl"
sparql_client = SparqlClient(TTL_FILE_PATH)


def semantic_search(request):
    query = request.GET.get("query", "").strip()
    mode = request.GET.get("mode", "search")

    column_names = {
        "Institution": "Институција",
        "institution": "Институција",
        "TotalContractValue": "Вкупна вредност на договори",
        "totalContractValue": "Вкупна вредност на договори",
        "Contract": "Договор",
        "contract": "Договор",
        "Amount": "Износ",
        "amount": "Износ",
        "Date": "Датум",
        "date": "Датум",
        "Year": "Година",
        "year": "Година",
        "ContractCount": "Број на договори",
        "contractCount": "Број на договори",
    }

    context = {
        "query": query,
        "mode": mode,
        "column_names": column_names,
    }

    if not query:
        return render(request, "semantic_search.html", context)

    try:
        gemini_response = ask_ai(query)
        sparql_query = gemini_response.get("sparql")
        results = sparql_client.query(sparql_query)

        context.update({
            "results": results,
            "results_count": len(results),
            "sparql_query": sparql_query,
        })

        if mode == "chat":
            context["conversational_response"] = (
                f"Пронајдовме {len(results)} резултати."
                if results else "📭 Нема резултати."
            )

    except Exception as e:
        context["error"] = str(e)

    return render(request, "semantic_search.html", context)


def chat_interface(request):
    return render(request, 'chat_interface.html')


@csrf_exempt
def conversational_chat(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        question = data.get("question", "").strip()
        mode = data.get("mode", "chat")
        if not question:
            return JsonResponse({"error": "No question provided"}, status=400)

        gemini_mode = "hybrid" if mode == "chat" else "sparql_only"
        gemini_response = ask_ai(question, mode=gemini_mode)
        gemini_mode_result = gemini_response.get("mode")

        if gemini_mode_result == "sparql":
            sparql_query = gemini_response.get("sparql")

            g = Graph()
            g.parse("kgapp/ontology/output.ttl", format="turtle")
            results = g.query(sparql_query)

            parsed_results = []
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
                parsed_results.append(item)

            if parsed_results:
                html_rows = ""
                for item in parsed_results:
                    row_str = ", ".join([f"<strong>{k}:</strong> {v}" for k, v in item.items()])
                    html_rows += f"<div>{row_str}</div>"
            else:
                html_rows = "📭 Нема пронајдени резултати."

            human_response = html_rows

            return JsonResponse({
                "response": human_response,
                "sparql_query": sparql_query,
                "results_count": len(parsed_results)
            })

        elif gemini_mode_result == "text":
            return JsonResponse({
                "response": gemini_response.get("answer"),
                "sparql_query": None,
                "results_count": None
            })

        else:
            return JsonResponse({"error": f"Unknown mode: {gemini_mode_result}"}, status=400)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
