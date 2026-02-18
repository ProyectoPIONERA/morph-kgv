from morph_kgc import VIRTStore
from rdflib import Graph

store = VIRTStore('testing/c.ini')

graph = Graph(store)

store

res = graph.query('''

PREFIX ub: <http://swat.cse.lehigh.edu/onto/univ-bench.owl#>

SELECT ?x ?y
WHERE {
  ?x a ub:Student .
  ?x ub:takesCourse ?y .
  ?y a ub:Course .
  <http://www.department0.university0.edu/associateProfessor0> ub:teacherOf ?y .
}

''')

res = graph.query('''

PREFIX ub: <http://swat.cse.lehigh.edu/onto/univ-bench.owl#>

SELECT ?x ?y ?z
WHERE {
  ?x a ub:GraduateStudent .
  ?x ub:undergraduateDegreeFrom ?y .
  ?x ub:memberOf ?z .
  ?y a ub:University .
  ?z a ub:Department .
  ?z ub:subOrganizationOf ?y .
}

''')

for row in res:
    pass
