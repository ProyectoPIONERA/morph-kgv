## Instalación

```
git clone https://github.com/ProyectoPIONERA/morph-kgv
cd morph-kgv
pip install .
```
Se recomienda utilizar entornos virtuales.

## Uso por línea de comandos

```
# Query from a .sparql / .rq file:
python run_query.py config.ini query.sparql

# Query passed directly as a string:
python run_query.py config.ini "SELECT ?x WHERE { ?x a <http://example.org/Thing> }"

# Read query from stdin:
python run_query.py config.ini -
```

El fichero de configuración `config.ini` es similar al de [morph-kgc](https://github.com/morph-kgc/morph-kgc). Un fichero de ejemplo está disponible en este repositorio.

## Despliegue de SPARQL endpoint

```
morph-kgv serve config.ini
```

El endpoint se despliega en `http://localhost:8000/sparql`.
