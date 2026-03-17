from morph_kgc import VIRTStore
from rdflib import Graph

store = VIRTStore('testing/c.ini')

graph = Graph(store)

'''
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
'''

res = graph.query('''

PREFIX ub: <http://swat.cse.lehigh.edu/onto/univ-bench.owl#>

SELECT ?x
WHERE {
  ?x a ub:Publication .
  ?x ub:publicationAuthor <http://www.department0.university0.edu/assistantProfessor0> .
}

''')


for row in res:
    pass
