#!/usr/bin/env python3
"""
main.py \u2014 VIRTStore SPARQL endpoint.

Serves a Morph-KGC mapping configuration as a SPARQL 1.1 endpoint backed by
VIRTStore, using rdflib-endpoint (FastAPI + YASGUI).

Usage
-----
  python main.py serve config.ini

  # Custom host / port
  python main.py serve config.ini --host 0.0.0.0 --port 9000

  # Disable Bloom-filter pre-join probe (B11) \u2014 useful for benchmarking
  python main.py serve config.ini --no-bloom

  # Enable per-step SQL debug traces
  python main.py serve config.ini --debug

  # Expose under a sub-path (e.g. behind a reverse proxy)
  python main.py serve config.ini --public-url https://example.org/sparql/

  # Enable hot-reload (development only)
  python main.py serve config.ini --reload

Then open http://<host>:<port> to use the YASGUI editor,
or send queries directly to http://<host>:<port>/sparql.

Installation
------------
  pip install "rdflib-endpoint[web]" click morph-kgc
"""

import os
import sys
from pathlib import Path

import click
import uvicorn
from rdflib import Graph
from .endpoint import SparqlEndpoint

# ---------------------------------------------------------------------------
# Default example query shown in the YASGUI editor tab
# ---------------------------------------------------------------------------
_DEFAULT_EXAMPLE_QUERY = """\
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT * WHERE {
    ?s ?p ?o .
} LIMIT 100
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """Serve a Morph-KGC mapping configuration as a SPARQL endpoint."""


@cli.command(help="Serve a Morph-KGC config.ini as a SPARQL endpoint via VIRTStore.")
@click.argument("config", metavar="CONFIG_INI")
@click.option("--host",       default="localhost", show_default=True, help="Bind host.")
@click.option("--port",       default=8000,        show_default=True, help="Bind port.", type=int)
@click.option("--public-url", default=None,        help="Public base URL (for reverse-proxy deployments).")
@click.option("--title",      default="VIRTStore SPARQL Endpoint", show_default=True, help="Endpoint title shown in the UI.")
@click.option("--reload",     is_flag=True, default=False, help="Enable uvicorn hot-reload (development only).")
def serve(
    config: str,
    host: str,
    port: int,
    public_url: "str | None",
    title: str,
    reload: bool,
) -> None:
    _run_serve(
        config=config,
        host=host,
        port=port,
        public_url=public_url,
        title=title,
        reload=reload,
    )


# ---------------------------------------------------------------------------
# Core logic (importable for testing / programmatic use)
# ---------------------------------------------------------------------------

def _run_serve(
    config: str,
    host: str = "localhost",
    port: int = 8000,
    public_url: "str | None" = None,
    title: str = "VIRTStore SPARQL Endpoint",
    bloom_filter: bool = True,
    debug: bool = False,
    reload: bool = False,
) -> None:
    """Initialise VIRTStore, build the FastAPI app, and start uvicorn."""

    # -- Enable VIRT_DEBUG before VIRTStore is imported --------------------
    # _VIRT_DEBUG is read at module-level inside virt_store.py, so the env
    # var must be set *before* the first import of morph_kgc.
    if debug:
        os.environ["VIRT_DEBUG"] = "1"
        click.echo(click.style("DEBUG", fg="yellow") + ": VIRT_DEBUG traces enabled")

    # -- Lazy import (respects the VIRT_DEBUG flag set above) --------------
    try:
        from morph_kgc import VIRTStore
    except ImportError as exc:
        msg = str(exc)
        click.echo(
            click.style("ERROR", fg="red")
            + ": could not import VIRTStore -- " + msg + "\n"
            + "  Make sure morph-kgc is installed: pip install morph-kgc"
        )
        sys.exit(1)

    # -- Validate config path ----------------------------------------------
    config_path = Path(config).resolve()
    if not config_path.exists():
        click.echo(
            click.style("ERROR", fg="red")
            + ": config file not found: " + config
        )
        sys.exit(1)

    # -- Build VIRTStore + RDFLib Graph ------------------------------------
    click.echo(
        click.style("INFO", fg="green")
        + ": Loading VIRTStore -> " + str(config_path)
    )
    store = VIRTStore(str(config_path), bloom_filter=bloom_filter)
    graph = Graph(store)

    bloom_status = "enabled" if bloom_filter else "disabled (--no-bloom)"
    click.echo(
        click.style("INFO", fg="green")
        + ": Bloom filter B11: " + bloom_status
    )

    # -- Resolve public URL ------------------------------------------------
    resolved_public_url = public_url or ("http://" + host + ":" + str(port) + "/")

    # -- Build SparqlEndpoint (FastAPI app) ---------------------------------
    app = SparqlEndpoint(
        graph=graph,
        path="/sparql",
        cors_enabled=True,
        title=title,
        description=(
            "A SPARQL 1.1 endpoint powered by **VIRTStore** (Morph-KGC virtual RDF graphs).\n\n"
            "Query your relational data sources using SPARQL without materialising the full graph.\n\n"
            "[Source code](https://github.com/vemonet/rdflib-endpoint)"
        ),
        version="1.0.0",
        public_url=resolved_public_url,
        enable_update=False,
        example_query=_DEFAULT_EXAMPLE_QUERY,
    )

    # -- Start server -------------------------------------------------------
    base_url = "http://" + host + ":" + str(port)
    click.echo(click.style("INFO", fg="green") + ": SPARQL endpoint ready at " + base_url)
    click.echo(click.style("INFO", fg="green") + ":   YASGUI editor  -> " + base_url)
    click.echo(click.style("INFO", fg="green") + ":   SPARQL service -> " + base_url + "/sparql")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(cli())
