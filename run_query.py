#!/usr/bin/env python3
"""
run_query.py — CLI wrapper for VIRTStore SPARQL queries.

Usage
-----
  # Query from a .sparql / .rq file:
  python run_query.py config.ini query.sparql

  # Query passed directly as a string:
  python run_query.py config.ini "SELECT ?x WHERE { ?x a <http://example.org/Thing> }"

  # Read query from stdin:
  python run_query.py config.ini -
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_query",
        description="Execute a SPARQL query against a VIRTStore mapping configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "config",
        metavar="CONFIG_INI",
        help="Path to the Morph-KGC config.ini file.",
    )
    parser.add_argument(
        "query",
        metavar="QUERY",
        help=(
            "SPARQL query: a file path (.sparql / .rq), "
            "a raw SPARQL string, or '-' to read from stdin."
        ),
    )
    parser.add_argument(
        "--no-bloom",
        action="store_true",
        default=False,
        help="Disable the Bloom-filter pre-join probe (B11). Useful for benchmarking.",
    )
    parser.add_argument(
        "--format",
        choices=["table", "csv", "tsv", "json"],
        default="table",
        help="Output format for SELECT results (default: table).",
    )
    parser.add_argument(
        "--timing-only",
        action="store_true",
        default=False,
        help="Suppress result rows; print only row count and elapsed time.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable VIRT_DEBUG logs (equivalent to VIRT_DEBUG=1 env var).",
    )
    return parser.parse_args()


def _load_query(query_arg: str) -> str:
    """Resolve the query argument to a SPARQL string."""
    if query_arg == "-":
        return sys.stdin.read()
    path = Path(query_arg)
    if path.suffix.lower() in (".sparql", ".rq") or path.exists():
        if not path.exists():
            print(f"Error: query file not found: {path}", file=sys.stderr)
            sys.exit(1)
        return path.read_text(encoding="utf-8")
    # Treat as a raw inline SPARQL string
    return query_arg


def _print_table(vars_: list, rows: list) -> None:
    """Pretty-print SELECT results as an aligned table."""
    str_rows = [[str(cell) if cell is not None else "" for cell in row] for row in rows]
    widths = [max(len(v), *(len(r[i]) for r in str_rows), 0) for i, v in enumerate(vars_)]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    header = "|" + "|".join(f" {v:<{w}} " for v, w in zip(vars_, widths)) + "|"
    print(sep)
    print(header)
    print(sep)
    for r in str_rows:
        print("|" + "|".join(f" {c:<{w}} " for c, w in zip(r, widths)) + "|")
    print(sep)


def _print_csv(vars_: list, rows: list, delimiter: str = ",") -> None:
    writer = csv.writer(sys.stdout, delimiter=delimiter)
    writer.writerow(vars_)
    for row in rows:
        writer.writerow([str(cell) if cell is not None else "" for cell in row])


def _print_json(vars_: list, rows: list) -> None:
    results = [
        {v: (str(cell) if cell is not None else None) for v, cell in zip(vars_, row)}
        for row in rows
    ]
    json.dump({"vars": vars_, "results": results}, sys.stdout, indent=2, ensure_ascii=False)
    print()  # trailing newline


def main() -> None:
    args = _parse_args()

    # Enable debug logging BEFORE importing virt_store (read at module load time)
    if args.debug:
        os.environ["VIRT_DEBUG"] = "1"

    # Lazy import so VIRT_DEBUG is set before the module is loaded
    from rdflib import Graph

    try:
        from morph_kgc import VIRTStore
    except ImportError as exc:
        print(f"Error: could not import VIRTStore — {exc}", file=sys.stderr)
        sys.exit(1)

    config_path = str(Path(args.config).resolve())
    if not Path(config_path).exists():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    sparql = _load_query(args.query)

    # Initialise store and graph
    store = VIRTStore(config_path, bloom_filter=not args.no_bloom)
    graph = Graph(store)

    # Execute query with timing
    start = time.perf_counter()
    try:
        result = graph.query(sparql)
    except Exception as exc:
        print(f"Error executing query: {exc}", file=sys.stderr)
        sys.exit(1)
    elapsed = time.perf_counter() - start

    # Collect rows (consuming the iterator once)
    rows = list(result)
    row_count = len(rows)

    if not args.timing_only:
        vars_ = [str(v) for v in result.vars] if result.vars else []

        if result.type == "SELECT":
            if args.format == "table":
                _print_table(vars_, rows)
            elif args.format == "csv":
                _print_csv(vars_, rows, delimiter=",")
            elif args.format == "tsv":
                _print_csv(vars_, rows, delimiter="\t")
            elif args.format == "json":
                _print_json(vars_, rows)
        elif result.type == "CONSTRUCT" or result.type == "DESCRIBE":
            for triple in rows:
                print(triple)
        elif result.type == "ASK":
            print(rows[0] if rows else False)

    print(f"\nRows: {row_count}  |  Elapsed: {elapsed * 1000:.3f} ms", file=sys.stderr)


if __name__ == "__main__":
    main()
