__author__ = "Julián Arenas-Guerrero"
__credits__ = ["Julián Arenas-Guerrero"]

__license__ = "Apache-2.0"
__maintainer__ = "Julián Arenas-Guerrero"
__email__ = "julian.arenas.guerrero@upm.es"


import duckdb
from typing import Iterable, List, Tuple, Set
from rdflib.store import Store
from rdflib.util import from_n3
from .types import Triple
from ..args_parser import load_config_from_argument
from ..mapping.mapping_parser import retrieve_mappings
import re
import pandas as pd
from rdflib import URIRef, Literal, BNode
from rdflib.term import Variable, Identifier
from rdflib.plugins.sparql.sparql import FrozenBindings, QueryContext
import rdflib.plugins.sparql.evaluate as sparql_evaluate
from rdflib.plugins.sparql.sparql import QueryContext
from rdflib.plugins.sparql import CUSTOM_EVALS
import rdflib.plugins.sparql as sp

from ..materializer import _materialize_mapping_group_to_df

_Triple = Tuple[Identifier, Identifier, Identifier]
BGP = List[_Triple]

from .types import BGP

import warnings
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)


# A triple pattern component is one of these rdflib types
RDFTerm = URIRef | Literal | BNode | Variable
TriplePattern = tuple[RDFTerm, RDFTerm, RDFTerm]




from typing import List, Any

def is_integer_string(s: str) -> bool:
    """True if s is a string representing a (signed) integer."""
    try:
        int(s.strip())
        return True
    except (ValueError, TypeError):
        return False

def keep_integer_strings_or_all(items: List[Any]) -> List[Any]:
    """
    Keep only integer strings if any exist; otherwise return the original list.
    Integer strings are strings that can be parsed by int() after stripping whitespace.
    """
    # Collect the integer-strings (preserve original string form)
    integers_only = [x for x in items if isinstance(x, str) and is_integer_string(x)]
    # If none found, return the original list unchanged
    return integers_only if integers_only else items




def triple_pattern_variables(triple_pattern: tuple) -> list[str]:
    """
    Extract the variable names from an rdflib triple pattern.

    Parameters
    ----------
    triple_pattern : tuple
        A 3-tuple (s, p, o) of rdflib terms.

    Returns
    -------
    list[str]
        Variable names (without the '?' prefix) in s→p→o order,
        with duplicates removed while preserving first-occurrence order.
    """
    seen = set()
    variables = []
    for term in triple_pattern:
        if isinstance(term, Variable):
            name = str(term)
            if name not in seen:
                seen.add(name)
                variables.append(name)
    return variables


def bgp_variables(bgp: list[tuple]) -> list[str]:
    seen = set()
    variables = []
    for triple in bgp:
        for var in triple_pattern_variables(triple):
            if var not in seen:
                seen.add(var)
                variables.append(var)
    return variables


def natural_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    tp,
    on: list[str] | None = None,
    how: str = "inner",
) -> pd.DataFrame:
    """
    Merge two DataFrames on shared columns.

    Parameters
    ----------
    left, right : pd.DataFrame
    on : list[str] | None
        Columns to join on. If None, auto-detects all overlapping columns.
        If an empty list is passed, returns the Cartesian product.
    how : str
        Join type: 'inner' (default), 'left', 'right', or 'outer'.

    Returns
    -------
    pd.DataFrame
    """
    if on is None:
        on = list(set(left.columns) & set(right.columns) & set(triple_pattern_variables(tp)))

    if not on:
        return pd.merge(left, right, how="cross")

    # Keep join keys from right, drop everything else already in left
    right_cols = on + [col for col in right.columns if col not in left.columns]
    return pd.merge(left, right[right_cols], on=on, how=how)


def rename_triple_columns(
    df: pd.DataFrame,
    triple_pattern: TriplePattern,
    drop_non_variables: bool = True,
) -> pd.DataFrame:
    """
    Rename 'subject', 'predicate', 'object' columns according to the
    Variable names in an rdflib triple pattern. Any extra columns in the
    DataFrame are left completely untouched.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain at least: subject, predicate, object.
        May contain any number of additional columns.
    triple_pattern : tuple[RDFTerm, RDFTerm, RDFTerm]
        A 3-tuple of rdflib terms. Positions holding a Variable are
        renamed; positions holding URIRef / Literal / BNode are concrete.
    drop_non_variables : bool, default False
        If True, drop the s/p/o columns whose position is a concrete term.
        Extra columns beyond s/p/o are never dropped.

    Returns
    -------
    pandas.DataFrame
        New DataFrame with relevant columns renamed (and optionally dropped).
        Column order is preserved.
    """
    if len(triple_pattern) != 3:
        raise ValueError("triple_pattern must be a 3-tuple (s, p, o)")

    # TODO: this line was added
    df = df[["subject", "predicate", "object"]]

    for col in ("subject", "predicate", "object"):
        if col not in df.columns:
            raise ValueError(f"DataFrame is missing required column: '{col}'")

    s, p, o = triple_pattern
    positions = (("subject", s), ("predicate", p), ("object", o))

    rename_map  = {}   # old_col → new_col  for Variable positions
    cols_to_drop = []  # old_col to remove  for concrete positions (if requested)

    for col, term in positions:
        if isinstance(term, Variable):
            rename_map[col] = str(term)        # Variable("person") → "person"
        elif drop_non_variables:
            cols_to_drop.append(col)

    result = df.rename(columns=rename_map)     # leaves all other columns intact

    if cols_to_drop:
        result = result.drop(columns=cols_to_drop)

    return result




# -----------------------------------------------------------------------------------------------------
# RML TEMPLATE AND RDF TERM MATCHING

def rml_template_to_regex(template: str) -> tuple[re.Pattern, list[str]]:
    """
    Convert an RML template string into a compiled regex
    and return the ordered list of reference names found in the template.

    Returns
    -------
    pattern : re.Pattern
        Regex with one named group per {reference}.
    references : list[str]
        Reference names in order of appearance.
    """
    parts = []
    references = []
    ref_count: dict[str, int] = {}
    i = 0

    while i < len(template):
        ch = template[i]

        if ch == '\\' and i + 1 < len(template):
            nxt = template[i + 1]
            if nxt in ('{', '}', '\\'):
                parts.append(re.escape(nxt))
                i += 2
                continue
            parts.append(re.escape('\\'))
            i += 1

        elif ch == '{':
            j = i + 1
            while j < len(template) and template[j] != '}':
                j += 1
            if j >= len(template):
                raise ValueError(f"Unclosed '{{' in RML template: {template!r}")

            ref_name = template[i + 1 : j]
            references.append(ref_name)

            # Regex group names must be valid Python identifiers
            safe = re.sub(r'\W', '_', ref_name)
            count = ref_count.get(safe, 0)
            ref_count[safe] = count + 1
            group_name = safe if count == 0 else f"{safe}_{count}"

            parts.append(f'(?P<{group_name}>.+?)')
            i = j + 1

        else:
            parts.append(re.escape(ch))
            i += 1

    return re.compile('^' + ''.join(parts) + '$'), references


def match_rml_template(rdf_term, template: str) -> dict[str, str] | None:
    """
    Match an RDF term against an RML template.

    Parameters
    ----------
    rdf_term : URIRef | Literal | BNode | str
        The RDF term to test.
    template : str
        The rr:template / rml:template string,
        e.g. "http://example.com/person/{ID}"

    Returns
    -------
    dict[str, str]
        Mapping {reference_name → matched_value} if the term matches.
    None
        If the term cannot have been generated by the template.
    """
    value = str(rdf_term)
    pattern, references = rml_template_to_regex(template)
    m = pattern.match(value)
    if m is None:
        return None

    groups = m.groupdict()
    ref_count: dict[str, int] = {}
    result: dict[str, str] = {}

    for ref_name in references:
        safe = re.sub(r'\W', '_', ref_name)
        count = ref_count.get(safe, 0)
        group_name = safe if count == 0 else f"{safe}_{count}"
        ref_count[safe] = count + 1

        if ref_name not in result:
            result[ref_name] = groups[group_name]
        else:
            result[f"{ref_name}#{count}"] = groups[group_name]

    return result


def match_triple_pattern(tp, rml_df):
    s_pat, p_pat, o_pat = tp

    # filter by term type
    if not isinstance(s_pat, Variable):
        if isinstance(s_pat, URIRef):
            rml_df = rml_df[rml_df["subject_termtype"] == 'http://w3id.org/rml/IRI']
        elif isinstance(s_pat, BNode):
            rml_df = rml_df[rml_df["subject_termtype"] == 'http://w3id.org/rml/BlankNode']

    if not isinstance(o_pat, Variable):
        if isinstance(o_pat, URIRef):
            rml_df = rml_df[rml_df["object_termtype"] == 'http://w3id.org/rml/IRI']
        elif isinstance(o_pat, Literal):
            rml_df = rml_df[rml_df["object_termtype"] == 'http://w3id.org/rml/Literal']
        elif isinstance(o_pat, BNode):
            rml_df = rml_df[rml_df["object_termtype"] == 'http://w3id.org/rml/BlankNode']


    # filter by term value
    rml_df.loc[:, "match"] = False
    for i, rule in rml_df.iterrows():
        s_match = p_match = o_match = False

        if isinstance(s_pat, Variable):
            s_match = True
        else:
            if str(rule['subject_map_type']) == 'http://w3id.org/rml/constant':
                if str(s_pat) == rule['subject_map_value']:
                    s_match = True
            elif str(rule['subject_map_type']) == 'http://w3id.org/rml/template':
                if match_rml_template(s_pat, str(rule['subject_map_value'])):
                    s_match = True

        if isinstance(p_pat, Variable):
            p_match = True
        else:
            if str(rule['predicate_map_type']) == 'http://w3id.org/rml/constant':
                if str(p_pat) == rule['predicate_map_value']:
                    p_match = True
            elif str(rule['predicate_map_type']) == 'http://w3id.org/rml/template':
                if match_rml_template(p_pat, str(rule['predicate_map_value'])):
                    p_match = True

        if isinstance(o_pat, Variable):
            o_match = True
        else:
            if str(rule['object_map_type']) == 'http://w3id.org/rml/constant':
                if str(o_pat) == rule['object_map_value']:
                    o_match = True
            elif str(rule['object_map_type']) == 'http://w3id.org/rml/template':
                if match_rml_template(o_pat, str(rule['object_map_value'])):
                    o_match = True
                    # TODO: also match for reference?
            #elif str(rule['object_map_type']) == 'http://w3id.org/rml/reference':
            #    o_match = True

        if s_match and p_match and o_match:
            rml_df.at[i, "match"] = True

    rml_df = rml_df[rml_df['match']]

    return rml_df




# -----------------------------------------------------------------------------------------------------
# BGP JOIN ORDERING FUNCTION


# Well-known low-selectivity predicates — extend as needed for your dataset
LOW_SELECTIVITY_PREDICATES: frozenset[str] = frozenset({
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
    "http://www.w3.org/2000/01/rdf-schema#domain",
    "http://www.w3.org/2000/01/rdf-schema#range",
    "http://www.w3.org/2002/07/owl#sameAs",
    "http://www.w3.org/2004/02/skos/core#broader",
    "http://www.w3.org/2004/02/skos/core#narrower",
})


def _position_weight(pos: int) -> int:
    return {0: 1, 2: 2, 1: 3}[pos]


def _is_bound(term, ctx, bound_vars):
    if not isinstance(term, Variable):
        return True
    return term in bound_vars or ctx[term] is not None


def _triple_score(triple, ctx, bound_vars):
    """
    Four-level score tuple — lower is better, compared lexicographically:

    Level 0 – connectivity     : 0 if shares ≥1 bound var (or first triple),
                                 1 if fully disconnected (bind-join safety)
    Level 1 – predicate penalty: 1 if predicate is a known low-selectivity URI,
                                 0 otherwise
    Level 2 – unbound count   : number of unbound positions (0–3)
    Level 3 – positional weight: sum of weights for unbound positions
                                 subject=1, object=2, predicate=3
    """
    s, p, o = triple

    # Level 0: connectivity (critical for bind joins)
    connected = any(_is_bound(t, ctx, bound_vars) and isinstance(t, Variable)
                    for t in triple)
    # If bound_vars is empty this is the first triple — treat as connected
    l0 = 0 if (not bound_vars or connected) else 1

    # Level 1: predicate penalty
    l1 = 1 if (isinstance(p, URIRef) and str(p) in LOW_SELECTIVITY_PREDICATES) else 0

    # Levels 2 & 3: unbound count + positional weights
    l2 = l3 = 0
    for pos, term in enumerate(triple):
        if not _is_bound(term, ctx, bound_vars):
            l2 += 1
            l3 += _position_weight(pos)

    return (l0, l1, l2, l3)


def order_bgp(ctx, bgp):
    """
    Reorder BGP triple patterns for efficient bind-join left-deep evaluation.

    Scoring per triple (lower = execute earlier):
      1. Connectivity  — disconnected patterns are always deferred
      2. Predicate     — known low-selectivity predicates (rdf:type etc.) are penalised
      3. Unbound count — fewer free variables → smaller intermediate result
      4. Positional    — subject(1) > object(2) > predicate(3) for remaining ties

    After each pick the chosen triple's variables are added to bound_vars
    (join propagation), so connected follow-on patterns score better next round.
    """
    if len(bgp) <= 1:
        return list(bgp)

    bound_vars = {
        t for triple in bgp for t in triple
        if isinstance(t, Variable) and ctx[t] is not None
    }

    remaining = list(bgp)
    ordered = []

    while remaining:
        best = min(range(len(remaining)),
                   key=lambda i: _triple_score(remaining[i], ctx, bound_vars))
        chosen = remaining.pop(best)
        ordered.append(chosen)
        for t in chosen:
            if isinstance(t, Variable):
                bound_vars.add(t)

    return ordered





# -----------------------------------------------------------------------------------------------------
# INJECT INTERMEDIATE BINDINGS IN SQL QUERIES


# ── RML vocabulary (http://w3id.org/rml/ namespace) ───────────────────────────
RML_TEMPLATE  = "http://w3id.org/rml/template"
RML_REFERENCE = "http://w3id.org/rml/reference"
RML_CONSTANT  = "http://w3id.org/rml/constant"
RML_QUERY     = "http://w3id.org/rml/query"
RML_TABLENAME = "http://w3id.org/rml/tableName"


def _template_to_regex_with_names(template: str) -> tuple[re.Pattern, list[str]]:
    """Compile an RML template string to a named-group regex + reference list."""
    parts, references, ref_count = [], [], {}
    i = 0
    while i < len(template):
        ch = template[i]
        if ch == '\\' and i + 1 < len(template) and template[i+1] in ('{', '}', '\\'):
            parts.append(re.escape(template[i+1])); i += 2
        elif ch == '{':
            j = i + 1
            while j < len(template) and template[j] != '}': j += 1
            ref = template[i+1:j]; references.append(ref)
            safe = re.sub(r'\W', '_', ref)
            count = ref_count.get(safe, 0); ref_count[safe] = count + 1
            parts.append(f'(?P<{safe if count == 0 else f"{safe}_{count}"}>.+?)'); i = j + 1
        else:
            parts.append(re.escape(ch)); i += 1
    return re.compile('^' + ''.join(parts) + '$'), references


def _extract_references_from_term(
    term, map_type: str, map_value: str
) -> dict[str, str] | None:
    """
    Reverse-engineer SQL column→value pairs from a bound RDF term.

    Returns {} if nothing can be pushed down (constants, unbound variables).
    Returns None if the term is structurally incompatible → caller prunes the rule.
    """
    if isinstance(term, Variable):
        return {}
    term_str = str(term)

    if map_type == RML_TEMPLATE:
        pattern, refs = _template_to_regex_with_names(map_value)
        m = pattern.match(term_str)
        if m is None:
            return None           # IRI doesn't match template → rule can be pruned
        ref_count, result = {}, {}
        for ref in refs:
            safe = re.sub(r'\W', '_', ref)
            count = ref_count.get(safe, 0); ref_count[safe] = count + 1
            result[ref] = m.group(safe if count == 0 else f"{safe}_{count}")
        return result

    elif map_type == RML_REFERENCE:
        return {map_value: term_str}   # column value = lexical form of the term

    elif map_type == RML_CONSTANT:
        return {}                      # fixed value, no column to push down

    return {}


def _wrap_existing_query(sql: str) -> str:
    """Wrap an existing SQL query as a subquery to safely append a WHERE clause."""
    return f"SELECT * FROM ({sql.strip().rstrip(';')}) AS _subquery"


def _inject_where(base_sql: str, conditions: list[str]) -> str:
    if not conditions:
        return base_sql
    cond_sql = " AND ".join(conditions)
    #if re.search(r'\bWHERE\b', base_sql, re.IGNORECASE):
    #    return f"{base_sql} AND {cond_sql}"
    return f"{base_sql} WHERE {cond_sql}"


def _build_conditions(ref_values: dict[str, list[str]]) -> list[str]:
    conditions = []
    for col, vals in ref_values.items():
        unique = list(dict.fromkeys(v for v in vals if v is not None))
        if not unique: continue

        # TODO: see LUBM4OBDA q12, it should be checked the datatypes of the filtered column so that value conditions with different datatype are removed from the condition
        # here if an integer is in the value conditions, integer datatype is assumes and non-integer strings are removed
        unique = keep_integer_strings_or_all(unique)
        escaped = [f"'{v}'" for v in unique]

        conditions.append(
            f"{col} = {escaped[0]}" if len(escaped) == 1
            else f"{col} IN ({', '.join(escaped)})"
        )
    return conditions


def pushdown_bindings_to_sql(
    triple_pattern: tuple,
    rml_df: pd.DataFrame,
    bindings_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Push down intermediate variable bindings into morph-kgc RML mapping SQL
    queries (bind join predicate pushdown).

    Parameters
    ----------
    triple_pattern : (s, p, o) of rdflib terms
    rml_df : morph-kgc internal mappings DataFrame
    bindings_df : intermediate variable bindings (columns = variable names)

    Returns
    -------
    Copy of rml_df with `logical_source_value` updated for matched rows.
    """
    s_pat, p_pat, o_pat = triple_pattern
    result_df = rml_df.copy()

    POSITIONS = [
        (s_pat, 'subject_map_type',   'subject_map_value'),
        (p_pat, 'predicate_map_type', 'predicate_map_value'),
        (o_pat, 'object_map_type',    'object_map_value'),
    ]
    has_bindings = (
        bindings_df is not None
        and not bindings_df.empty
    )

    for idx, row in result_df.iterrows():
        if row.get('source_type') != 'RDB':
            continue

        ref_values: dict[str, list[str]] = {}
        skip_row = False

        for pat_term, type_col, value_col in POSITIONS:
            map_type  = str(row.get(type_col,  '') or '')
            map_value = str(row.get(value_col, '') or '')

            if isinstance(pat_term, Variable):
                var_name = str(pat_term)
                # TODO
                if has_bindings and var_name in bindings_df.columns:
                    for term_val in bindings_df[var_name].dropna().unique():
                        refs = _extract_references_from_term(term_val, map_type, map_value)
                        if refs is None:
                            skip_row = True; break
                        for ref, val in refs.items():
                            ref_values.setdefault(ref, []).append(val)
                if skip_row: break
            else:
                refs = _extract_references_from_term(pat_term, map_type, map_value)
                if refs is None:
                    skip_row = True; break
                for ref, val in refs.items():
                    ref_values.setdefault(ref, []).append(val)

        if skip_row or not ref_values:
            continue

        conditions = _build_conditions(ref_values)
        if not conditions:
            continue

        src_type = str(row.get('logical_source_type', '') or '')
        src_val  = str(row.get('logical_source_value', '') or '').strip()

        if src_type == RML_QUERY:
            base_sql = _wrap_existing_query(src_val)
        elif src_type == RML_TABLENAME:
            base_sql = f"SELECT * FROM {src_val}"
            result_df.at[idx, 'logical_source_type'] = RML_QUERY # not tablename anymore
        else:
            continue

        result_df.at[idx, 'logical_source_value'] = _inject_where(base_sql, conditions)

    return result_df



# -----------------------------------------------------------------------------------------------------

def virt_eval_bgp(ctx: QueryContext, bgp: BGP, rml_df, config):

    for tp in bgp:
        print(str(tp[0]), str(tp[1]), str(tp[2]))


    # first tp in BGP
    first_tp = bgp[0]
    rml_tp_df = match_triple_pattern(first_tp, rml_df)
    print("TP:", first_tp[0], first_tp[1], first_tp[2])
    print("Number of matched rules:", len(rml_tp_df))
    #rml_tp_df.to_csv('x.csv', index=False)

    rml_tp_df = pushdown_bindings_to_sql((first_tp[0], first_tp[1], first_tp[2]), rml_tp_df, None)
    for i, row in rml_tp_df.iterrows():
        #print(row['logical_source_value'], '\n')
        pass

    data = _materialize_mapping_group_to_df(rml_tp_df, rml_df, None, config)
    data = rename_triple_columns(data, (first_tp[0], first_tp[1], first_tp[2]))
    bindings_df = data
    print('\n--------------------------------------------------------------\n')

    # execute BGP
    for tp in bgp[1:]:
        rml_tp_df = match_triple_pattern(tp, rml_df)
        rml_tp_df = pushdown_bindings_to_sql((tp[0], tp[1], tp[2]), rml_tp_df, bindings_df)
        for i, row in rml_tp_df.iterrows():
            print(row['logical_source_value'], '\n')
        print("TP:", str(tp[0]), str(tp[1]), str(tp[2]))
        print("Number of matched rules:", len(rml_tp_df))

        data = _materialize_mapping_group_to_df(rml_tp_df, rml_df, None, config)
        # RENAME DF TO VARIABLE NAMES
        data = rename_triple_columns(data, (tp[0], tp[1], tp[2]))
        #print(bindings_df.columns)

        # MERGE DF TO INTERMEDIATE DF
        bindings_df = natural_join(bindings_df, data, (tp[0], tp[1], tp[2]))

        print('\n--------------------------------------------------------------\n')

    """
    print(len(bindings_df))
    bindings_df = bindings_df[bgp_variables(bgp)]
    for i, row in bindings_df.iterrows():
        res = ''
        for var in bgp_variables(bgp):
            res += row[var] + ' '
        print(res)
    """

    # yield/return the results
    print(bindings_df)
    for i, row in bindings_df.iterrows():
        # convert the bindings_df into a FrozenBindings object
        bindings = dict()
        for key in bindings_df.columns:
            bindings[Variable(key)] = row[key]
        yield FrozenBindings(ctx, bindings)
    return




# -----------------------------------------------------------------------------------------------------
# RDFLIB QUERY EXECUTION

class VIRTStore(Store):

    context_aware = False   # set True if your backend supports named graphs
    formula_aware = False
    transaction_aware = False

    _EVAL_KEY = "VIRTStore"  # unique key in CUSTOM_EVALS

    def __init__(self, config_path: str, configuration=None, identifier=None):
        super(VIRTStore, self).__init__(configuration=configuration, identifier=identifier)

        config = load_config_from_argument(config_path)

        rml_df, fnml_df, http_api_df = retrieve_mappings(config)
        config.set('CONFIGURATION', 'http_api_df', http_api_df.to_csv())
        # keep only asserted mapping rules
        rml_df = rml_df.loc[rml_df['triples_map_type'] == 'http://w3id.org/rml/TriplesMap']

        self.rml_df = rml_df
        self.config = config

        # Register custom_eval — bind it to this instance with a closure
        CUSTOM_EVALS[self._EVAL_KEY] = self._make_custom_eval()

    # ------------------------------------------------------------------ #
    # Store lifecycle                                                       #
    # ------------------------------------------------------------------ #

    #def open(self, configuration: str, create: bool = False) -> Optional[int]:
        #    # connect to your backend here
        #    self._connection = self._connect(configuration)
        #    if self._connection is None:
        #        return NO_STORE
        # Register custom_eval — bind it to this instance with a closure
        #    CUSTOM_EVALS[self._EVAL_KEY] = self._make_custom_eval()
    #    return VALID_STORE

    #def close(self, commit_pending_transaction: bool = False) -> None:
        #    CUSTOM_EVALS.pop(self._EVAL_KEY, None)
        #    if self._connection:
        #        self._connection.close()
    #        self._connection = None

    # ------------------------------------------------------------------ #
    # Custom eval factory — returns a closure that captures `self`         #
    # ------------------------------------------------------------------ #

    def _make_custom_eval(self):
        #store = self  # capture the store instance

        def custom_eval(ctx, part):
            if part.name == "BGP":
                ordered = order_bgp(ctx, part.triples)
                return virt_eval_bgp(ctx, ordered, self.rml_df, self.config)
            # Any other algebra node (FILTER, LeftJoin/OPTIONAL,
            # UNION, Extend/BIND, etc.) falls back to RDFLib natively.
            raise NotImplementedError()

        return custom_eval


    # ------------------------------------------------------------------ #
    # Mandatory Store interface methods                                    #
    # ------------------------------------------------------------------ #

    def triples(self, pattern, context) -> Iterable[Triple]:
        def triples(self, triple_pattern, context=None) -> Iterator[Tuple]:
            # Used by rdflib for Graph iteration — can delegate to _query_triple
            s, p, o = triple_pattern
            for row in self._query_triple(s, p, o):
                yield row, iter([context])

    def create(self, configuration):
        raise TypeError('The VIRT store is read only!')

    def destroy(self, configuration):
        raise TypeError('The VIRT store is read only!')

    def commit(self):
        raise TypeError('The VIRT store is read only!')

    def rollback(self):
        raise TypeError('The VIRT store is read only!')

    def add(self, _, context=None, quoted=False):
        raise TypeError('The VIRT store is read only!')

    def addN(self, quads):
        raise TypeError('The VIRT store is read only!')

    def remove(self, _, context):
        raise TypeError('The VIRT store is read only!')
