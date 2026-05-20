#!/usr/bin/env python3
"""
query_endpoint.py — SPARQL client for the VIRTStore endpoint.

Sends queries to a running SPARQL 1.1 endpoint (e.g. started with main.py)
using the SPARQLWrapper library. Supports SELECT, ASK, CONSTRUCT and DESCRIBE.

Usage
-----
  # Query from a .sparql / .rq file
  python query_endpoint.py query.sparql

  # Inline SPARQL string
  python query_endpoint.py "SELECT * WHERE { ?s ?p ?o } LIMIT 10"

  # Read from stdin
  cat query.sparql | python query_endpoint.py -

  # Custom endpoint URL
  python query_endpoint.py query.sparql --endpoint http://0.0.0.0:9000/sparql

  # Output formats: table (default), csv, tsv, json
  python query_endpoint.py query.sparql --format csv

  # Suppress result rows, print timing only
  python query_endpoint.py query.sparql --timing-only

Installation
------------
  pip install sparqlwrapper click
"""

import csv
import json
import sys
import time
from pathlib import Path

import click
from SPARQLWrapper import JSON, POST, GET, TURTLE, SPARQLWrapper
from SPARQLWrapper.SPARQLExceptions import EndPointNotFound, QueryBadFormed, SPARQLWrapperException

_DEFAULT_ENDPOINT = "http://localhost:8000/sparql"


# ---------------------------------------------------------------------------
# Query loading
# ---------------------------------------------------------------------------

def _load_query(query_arg: str) -> str:
    if query_arg == "-":
        return sys.stdin.read()
    path = Path(query_arg)
    if path.suffix.lower() in (".sparql", ".rq") or (path.exists() and path.is_file()):
        if not path.exists():
            click.echo(click.style("ERROR", fg="red") + ": query file not found: " + query_arg, err=True)
            sys.exit(1)
        return path.read_text(encoding="utf-8")
    return query_arg


def _detect_query_type(sparql: str) -> str:
    """Detect SELECT / ASK / CONSTRUCT / DESCRIBE from the query string."""
    stripped = sparql.strip().upper()
    # Skip PREFIX / BASE declarations before checking the query form
    for line in stripped.splitlines():
        line = line.strip()
        if not line or line.startswith("PREFIX") or line.startswith("BASE") or line.startswith("#"):
            continue
        for keyword in ("SELECT", "ASK", "CONSTRUCT", "DESCRIBE"):
            if line.startswith(keyword):
                return keyword
        break
    # Fallback: scan the whole string
    for keyword in ("SELECT", "ASK", "CONSTRUCT", "DESCRIBE"):
        if keyword in stripped:
            return keyword
    return "SELECT"


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _print_table(headers: list, rows: list) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows), 0) for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    def fmt_row(cells):
        return "|" + "|".join(" " + c.ljust(w) + " " for c, w in zip(cells, widths)) + "|"
    click.echo(sep)
    click.echo(fmt_row(headers))
    click.echo(sep)
    for row in rows:
        click.echo(fmt_row(row))
    click.echo(sep)


def _print_csv(headers: list, rows: list, delimiter: str = ",") -> None:
    writer = csv.writer(sys.stdout, delimiter=delimiter)
    writer.writerow(headers)
    for row in rows:
        writer.writerow(row)


def _print_json_select(headers: list, rows: list) -> None:
    results = [dict(zip(headers, row)) for row in rows]
    json.dump({"vars": headers, "results": results}, sys.stdout, indent=2, ensure_ascii=False)
    click.echo()


# ---------------------------------------------------------------------------
# Result handlers
# ---------------------------------------------------------------------------

def _handle_select(results: dict, fmt: str, timing_only: bool) -> int:
    # SPARQL 1.1 JSON format: { "head": { "vars": [...] }, "results": { "bindings": [...] } }
    vars_ = results.get("head", {}).get("vars", [])
    bindings = results.get("results", {}).get("bindings", [])
    rows = [
        [b.get(v, {}).get("value", "") for v in vars_]
        for b in bindings
    ]
    count = len(rows)
    if not timing_only:
        if fmt == "table":
            _print_table(vars_, rows)
        elif fmt == "csv":
            _print_csv(vars_, rows, ",")
        elif fmt == "tsv":
            _print_csv(vars_, rows, "\t")
        elif fmt == "json":
            _print_json_select(vars_, rows)
    return count


def _handle_ask(results: dict, timing_only: bool) -> int:
    answer = results.get("boolean", False)
    if not timing_only:
        click.echo("true" if answer else "false")
    return 1


def _handle_construct_describe(raw_text: str, timing_only: bool) -> int:
    lines = [l for l in raw_text.strip().splitlines() if l.strip()]
    if not timing_only:
        click.echo(raw_text.strip())
    return len(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument("query", metavar="QUERY")
@click.option(
    "--endpoint", "-e",
    default=_DEFAULT_ENDPOINT,
    show_default=True,
    help="SPARQL endpoint URL.",
)
@click.option(
    "--method",
    type=click.Choice(["GET", "POST"], case_sensitive=False),
    default="POST",
    show_default=True,
    help="HTTP method for the SPARQL request.",
)
@click.option(
    "--format", "-f", "fmt",
    type=click.Choice(["table", "csv", "tsv", "json"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format for SELECT results.",
)
@click.option(
    "--timeout", "-t",
    default=120,
    show_default=True,
    type=int,
    help="Request timeout in seconds.",
)
@click.option(
    "--timing-only",
    is_flag=True,
    default=False,
    help="Suppress result rows; print only row count and elapsed time.",
)
def main(
    query: str,
    endpoint: str,
    method: str,
    fmt: str,
    timeout: int,
    timing_only: bool,
) -> None:
    """Send a SPARQL query to the VIRTStore endpoint and display results.

    QUERY can be a .sparql/.rq file path, an inline SPARQL string, or - to read from stdin.
    """
    sparql_str = _load_query(query)
    query_type = _detect_query_type(sparql_str)

    # -- Configure SPARQLWrapper -------------------------------------------
    wrapper = SPARQLWrapper(endpoint)
    wrapper.setQuery(sparql_str)
    wrapper.setTimeout(timeout)
    wrapper.setMethod(POST if method.upper() == "POST" else GET)

    # SELECT / ASK -> JSON;  CONSTRUCT / DESCRIBE -> Turtle
    if query_type in ("SELECT", "ASK"):
        wrapper.setReturnFormat(JSON)
    else:
        wrapper.setReturnFormat(TURTLE)

    click.echo(
        click.style("INFO", fg="green")
        + ": Sending " + query_type + " to " + endpoint,
        err=True,
    )

    # -- Execute with timing -----------------------------------------------
    start = time.perf_counter()
    try:
        raw = wrapper.query()
    except EndPointNotFound:
        click.echo(
            click.style("ERROR", fg="red")
            + ": Endpoint not found — is the server running at " + endpoint + "?",
            err=True,
        )
        sys.exit(1)
    except QueryBadFormed as exc:
        click.echo(click.style("ERROR", fg="red") + ": Bad query — " + str(exc), err=True)
        sys.exit(1)
    except SPARQLWrapperException as exc:
        click.echo(click.style("ERROR", fg="red") + ": SPARQL error — " + str(exc), err=True)
        sys.exit(1)
    elapsed = time.perf_counter() - start

    # -- Parse and display -------------------------------------------------
    count = 0
    if query_type in ("SELECT", "ASK"):
        results = raw.convert()
        if query_type == "SELECT":
            count = _handle_select(results, fmt, timing_only)
        else:
            count = _handle_ask(results, timing_only)
    else:
        count = _handle_construct_describe(raw.response.read().decode("utf-8"), timing_only)

    click.echo(
        "\nRows: " + str(count) + "  |  Elapsed: " + f"{elapsed * 1000:.3f}" + " ms",
        err=True,
    )


if __name__ == "__main__":
    main()
