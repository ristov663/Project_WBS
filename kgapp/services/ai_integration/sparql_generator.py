import re
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY is not set")
genai.configure(api_key=api_key)


def extract_sparql_from_code_block(text: str) -> str:
    match = re.search(r"``````", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


SPARQL_BASE_PROMPT = """
You are an expert SPARQL query generator for a public procurement RDF ontology.

Always use this prefix:
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>

Classes:
- :Contract (but NOT declared with rdf:type)
- :Institution
- :Supplier

Properties:
- :hasAmount (float)
- :hasDate (date)
- :hasInstitution (Institution)
- :hasSupplier (Supplier)
- :hasDescription (string)

IMPORTANT: Contracts do NOT have an explicit rdf:type. Never use ?contract a :Contract .
Always identify ?contract by matching on its properties (e.g., ?contract :hasAmount ?amount).
For dates, always extract years using: BIND(YEAR(?date) AS ?year)
Use GROUP BY, ORDER BY and LIMIT when needed.
Use clear variable names.
Return ONLY valid SPARQL — no markdown or code fences.

Examples of GOOD patterns:

-- Top 5 contracts by amount:
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?contract ?amount
WHERE {{
  ?contract :hasAmount ?amount .
}}
ORDER BY DESC(?amount)
LIMIT 5

-- Number of contracts per year:
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?year (COUNT(?contract) AS ?contractCount)
WHERE {{
  ?contract :hasDate ?date .
  BIND(YEAR(?date) AS ?year)
}}
GROUP BY ?year
ORDER BY ?year

-- Top 5 institutions by total contract amount:
PREFIX : <http://www.semanticweb.org/pc/ontologies/2025/1/untitled-ontology-19/>
SELECT ?institution (SUM(?amount) AS ?totalAmount)
WHERE {{
  ?contract :hasInstitution ?institution ;
            :hasAmount ?amount .
}}
GROUP BY ?institution
ORDER BY DESC(?totalAmount)
LIMIT 5
""".strip()


def ask_ai(question: str, mode: str = "sparql_only") -> dict:
    model = genai.GenerativeModel("gemini-2.5-flash")

    if mode == "sparql_only":
        prompt = f"""
{SPARQL_BASE_PROMPT}

---
Now convert this question to SPARQL:
{question}
        """.strip()

        response = model.generate_content([{"role": "user", "parts": [prompt]}])
        sparql_query = extract_sparql_from_code_block(response.text.strip())
        return {
            "mode": "sparql",
            "sparql": sparql_query,
            "answer": None
        }

    elif mode == "hybrid":
        prompt = f"""
{SPARQL_BASE_PROMPT}

** Additional instructions for conversation mode **
- If the question needs RDF data, generate SPARQL.
- If it does NOT, just answer in text.
- Always respond in JSON:
{{
  "mode": "sparql" or "text",
  "sparql": "...",
  "answer": "..."
}}

Examples:
Question: "What is public procurement?"
Answer: {{
  "mode": "text",
  "sparql": "",
  "answer": "Public procurement means how governments buy goods and services..."
}}

---
Now answer:
Question: "{question}"
        """.strip()

        response = model.generate_content([{"role": "user", "parts": [prompt]}])
        print("\n[Gemini Raw Response]\n", response.text)

        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            raise ValueError("Could not parse Gemini response as JSON")

    else:
        raise ValueError(f"Invalid mode: {mode}")
