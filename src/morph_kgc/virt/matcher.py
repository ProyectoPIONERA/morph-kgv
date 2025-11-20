from rdflib import Graph, Namespace
import re

# === Namespaces ===
RML = Namespace("http://w3id.org/rml/")
RR  = Namespace("http://www.w3.org/ns/r2rml#")
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")

OUTPUT_FILE = "resultado_final_matching.txt"


# =========================================================
#   QUERY ÚNICA — devuelve toda la info necesaria
# =========================================================
QUERY = """
SELECT DISTINCT ?tm ?predicate ?predicateValue ?valueType ?object ?template WHERE { 
    ?tm rml:predicateObjectMap ?pom .
    
    ?pom rml:predicateMap ?pm .
    ?pm ?pType ?predicate .
    ?pm ?valueType ?predicateValue .

    OPTIONAL {
        ?pom rml:objectMap ?om .
        ?om ?valueType ?object .
    }

    OPTIONAL {
        ?tm rml:subjectMap ?sm .
        ?sm rml:template ?template .
    }

    FILTER(?pType IN (rml:constant, rml:template, rml:reference))
    FILTER(?valueType IN (rml:constant, rml:template, rml:reference))
}
"""


# =========================================================
#   Helpers
# =========================================================
def template_to_example_uri(template: str) -> str:
    return re.sub(r"\{[^}]+\}", "1", template)


def match_uri_to_template(uri: str, template: str) -> bool:
    regex = re.escape(template)
    regex = re.sub(r"\\\{[^}]+\\\}", r"(.+)", regex)
    return re.fullmatch(regex, uri) is not None



# =========================================================
#   MAIN
# =========================================================
if __name__ == "__main__":

    graph = Graph()
    graph.parse("example.ttl", format="turtle")

    # Ejecutar solo UNA consulta
    rows = list(graph.query(QUERY, initNs={"rml": RML, "rr": RR}))

    # ---- Preparar estructuras reutilizables ----
    predicate_to_tm = []
    subject_templates = {}
    predicate_objects = {}

    for row in rows:
        tm = str(row.tm)
        predicate = str(row.predicate)
        predicate_value = str(row.predicateValue)
        template = str(row.template) if row.template else None
        object_val = str(row.object) if row.object else None
        value_type = str(row.valueType)

        # ---- 1) Predicates con TriplesMap ----
        predicate_to_tm.append({"triplesMap": tm, "predicate": predicate_value})

        # ---- 2) Subject templates ----
        if template:
            subject_templates[tm] = template

        # ---- 3) Group predicate-object ----
        if predicate not in predicate_objects:
            predicate_objects[predicate] = []

        if object_val:
            predicate_objects[predicate].append({
                "object": object_val,
                "type": value_type.split("#")[-1] if "#" in value_type else value_type
            })


    # =========================================================
    #   ESCRITURA EN ARCHIVO
    # =========================================================
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        # ---- SECTION 1: Predicates + TMs ----
        f.write("🔍 Predicados y TriplesMap asociados\n")
        f.write("===================================\n\n")

        for entry in predicate_to_tm:
            f.write(f" ✔ TriplesMap: {entry['triplesMap']}\n")
            f.write(f"    ↳ Predicate: {entry['predicate']}\n\n")


        # ---- SECTION 2: Subject Templates ----
        f.write("\n🔍 Subject Templates detectadas\n")
        f.write("================================\n\n")

        for tm, template in subject_templates.items():
            f.write(f" ✔ {tm} → {template}\n")


        # ---- SECTION 3: Matching Subject Templates ----
        f.write("\n\n🔎 Matching automático entre Subject Templates\n")
        f.write("==============================================\n")

        templates_list = list(subject_templates.items())

        for tm, template in templates_list:

            test_uri = template_to_example_uri(template)
            f.write(f"\n➡ URI generada: `{test_uri}` (desde: {template})\n")

            for tm2, t2 in templates_list:
                if match_uri_to_template(test_uri, t2):
                    f.write(f"   ✓ MATCH → {t2} (TM: {tm2})\n")
                else:
                    f.write(f"   ✗ NO MATCH → {t2} (TM: {tm2})\n")


        # ---- SECTION 4: Predicate → Object group ----
        f.write("\n\n📌 Predicates y sus ObjectMaps asociados\n")
        f.write("===========================================\n\n")

        for predicate, objects in predicate_objects.items():
            f.write(f"🔹 Predicate: {predicate}\n")

            for obj in objects:
                f.write(f"   • {obj['object']}   ({obj['type']})\n")

            f.write("\n----------------------------------------------------\n\n")


    print(f"\n📁 Archivo generado correctamente en: {OUTPUT_FILE}\n")
