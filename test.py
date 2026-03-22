from morph_kgc import VIRTStore
from rdflib import Graph

store = VIRTStore('testing/c.ini')

graph = Graph(store)

res = graph.query('''
PREFIX ub: <http://swat.cse.lehigh.edu/onto/univ-bench.owl#>

SELECT ?x ?y ?z
WHERE {
  ?x a ub:GraduateStudent .
  ?x ub:undergraduateDegreeFrom ?y .
  ?x ub:memberOf ?z .
  ?y a ub:University .
  OPTIONAL { ?z a ub:Department .
  ?z ub:subOrganizationOf ?y . }
}
''')

print(len(res))
for row in res:
    print(row)
