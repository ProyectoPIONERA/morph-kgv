from morph_kgc import VIRTStore
from rdflib import Graph
import time


"""
# Default — B11 enabled
store = VIRTStore("config.ini")

# Disabled — useful for benchmarking
store = VIRTStore("config.ini", bloom_filter=False)

# Tune thresholds at module level if needed
import morph_kgc.sparql.virt_store as vs
vs._B11_THRESHOLD  = 128   # activate only for larger intermediate sets
vs._B11_BIT_SIZE   = 1 << 18  # 256 KB — lower false-positive rate
vs._B11_HASH_COUNT = 5
"""




store = VIRTStore('testing/c.ini')

graph = Graph(store)

start = time.perf_counter()

res = graph.query('''
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX gtfs: <http://vocab.gtfs.org/terms#>
PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
PREFIX dct: <http://purl.org/dc/terms/>

SELECT * WHERE {
	?shape a gtfs:Shape .
	?shape gtfs:shapePoint ?shapePoint .
	?shapePoint geo:lat ?shape_pt_lat .
	?shapePoint geo:long ?shape_pt_lon .
	?shapePoint gtfs:pointSequence ?shape_pt_sequence .
}
''')


for row in res:
    print(row)
print(len(res))


elapsed = time.perf_counter() - start
print(f"Elapsed: {elapsed*1000:.3f} ms")
