from morph_kgc import VIRTStore
from rdflib import Graph
import time


"""
# VIRT_DEBUG=1 python your_script.py
# semijoin_reduction: bool = False

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
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX gtfsaccessible: <http://transport.linkeddata.es/resource/WheelchairBoardingStatus/>
PREFIX gtfslocation: <http://transport.linkeddata.es/resource/LocationType/>

SELECT ?longName (count(?name) as ?count)
WHERE { 	
	?route a gtfs:Route .
	?route gtfs:longName ?longName .
 
	?trip a gtfs:Trip .
	?trip gtfs:route ?route .
 
	?stopTime a gtfs:StopTime .
	?stopTime gtfs:trip ?trip .
	?stopTime gtfs:stop ?stop .
 
 	?stop a gtfs:Stop .
 	?stop foaf:name  ?name .

 	?stop gtfs:wheelchairAccessible gtfsaccessible:1 .	
 	
} GROUP BY ?longName
''')


for row in res:
    print(row)
print(len(res))


elapsed = time.perf_counter() - start
print(f"Elapsed: {elapsed*1000:.3f} ms")
