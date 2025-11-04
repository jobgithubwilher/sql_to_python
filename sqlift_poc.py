#!/usr/bin/env python3
"""
SQLift PoC: Simple SQL → pandas code generator for Dataiku.
- Supports: SELECT, FROM, simple INNER JOIN ... ON a=b, WHERE, GROUP BY,
  ORDER BY, LIMIT
- Output: readable pandas code you can paste into a Dataiku Python recipe.
- Usage:
    python sqlift_poc.py --sql example_input.sql \
        --table orders=df_orders \
        --table customers=df_customers \
        --out generated_output.py
"""

import argparse
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple


class SQLQuery:
    """Structured representation of the parts of a parsed SQL statement."""

    def __init__(self):
        self.select: List[str] = []
        self.from_table: Optional[str] = None
        self.joins: List[Dict[str, Any]] = []
        self.where: Optional[str] = None
        self.having: Optional[str] = None
        self.group_by: List[str] = []
        self.order_by: List[str] = []
        self.limit: Optional[int] = None


def _normalize_whitespace(s: str) -> str:
    """Collapse whitespace so regex parsing can target single-line SQL fragments."""
    return " ".join(s.strip().replace("\n", " ").split())


def parse_sql(sql: str) -> SQLQuery:
    """Parse a limited SQL select statement into an intermediate SQLQuery object."""
    sql = _normalize_whitespace(sql)
    q = SQLQuery()

    m = re.search(r"select\s+(.*?)\s+from\s+", sql, re.IGNORECASE)
    if m:
        q.select = [x.strip() for x in m.group(1).split(",")]

    m = re.search(r"from\s+([a-zA-Z0-9_\.]+)", sql, re.IGNORECASE)
    if m:
        q.from_table = m.group(1)

    for j in re.finditer(
        r"join\s+([a-zA-Z0-9_\.]+)(?:\s+(?:as\s+)?[a-zA-Z0-9_\.]+)?\s+on\s+([^\s]+)\s*=\s*([^\s]+)",
        sql,
        re.IGNORECASE,
    ):
        q.joins.append({"table": j.group(1), "left": j.group(2), "right": j.group(3)})

    m = re.search(
        r"where\s+(.*?)\s*(group by|having|order by|limit|$)", sql, re.IGNORECASE
    )
    if m:
        q.where = m.group(1)

    m = re.search(r"group by\s+(.*?)\s*(having|order by|limit|$)", sql, re.IGNORECASE)
    if m:
        q.group_by = [x.strip() for x in m.group(1).split(",")]

    m = re.search(r"having\s+(.*?)\s*(order by|limit|$)", sql, re.IGNORECASE)
    if m:
        q.having = m.group(1).strip()

    m = re.search(r"order by\s+(.*?)\s*(limit|$)", sql, re.IGNORECASE)
    if m:
        entries = []
        for part in m.group(1).split(","):
            cleaned = part.strip().rstrip(";")
            if cleaned:
                entries.append(cleaned)
        q.order_by = entries

    m = re.search(r"limit\s+(\d+)", sql, re.IGNORECASE)
    if m:
        q.limit = int(m.group(1))

    return q


def _strip_table_prefixes(expr: str) -> str:
    """Drop table aliases like alias.column without touching quoted strings."""
    result: List[str] = []
    i = 0
    in_single = False
    in_double = False
    while i < len(expr):
        ch = expr[i]
        if ch == "'" and not in_double:
            in_single = not in_single
            result.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            result.append(ch)
            i += 1
            continue
        if not in_single and not in_double and (ch.isalpha() or ch == "_"):
            j = i
            while j < len(expr) and (expr[j].isalnum() or expr[j] in "._"):
                j += 1
            token = expr[i:j]
            if "." in token:
                result.append(token.split(".")[-1])
            else:
                result.append(token)
            i = j
        else:
            result.append(ch)
            i += 1
    return "".join(result)


_MONTH_MAP = {
    "JAN": "01",
    "FEB": "02",
    "MAR": "03",
    "APR": "04",
    "MAY": "05",
    "JUN": "06",
    "JUL": "07",
    "AUG": "08",
    "SEP": "09",
    "OCT": "10",
    "NOV": "11",
    "DEC": "12",
}


def _convert_sas_date_literals(expr: str) -> str:
    """Convert SAS style date literals ('01JAN2024'd) into ISO strings."""

    def repl(match: re.Match[str]) -> str:
        day, mon, year = match.group(1), match.group(2).upper(), match.group(3)
        iso = f"{year}-{_MONTH_MAP.get(mon, '01')}-{int(day):02d}"
        return f"'{iso}'"

    return re.sub(r"'(\d{1,2})([A-Za-z]{3})(\d{4})'d", repl, expr)


def _expand_between_conditions(expr: str) -> str:
    """Re-write BETWEEN expressions into explicit greater-than/less-than comparisons."""
    pattern = re.compile(r"(?i)([A-Za-z_][\w\.]*)\s+BETWEEN\s+([^ ]+)\s+AND\s+([^ ]+)")

    def repl(match: re.Match[str]) -> str:
        col = match.group(1)
        lower = match.group(2)
        upper = match.group(3).rstrip(";")
        return f"({col} >= {lower}) and ({col} <= {upper})"

    return pattern.sub(repl, expr)


def _convert_in_clauses(expr: str) -> str:
    """Translate IN (...) syntax into pandas-friendly in/not in expressions."""
    pattern = re.compile(r"(?i)([A-Za-z_][\w\.]*)\s+(NOT\s+)?IN\s*\(([^)]+)\)")

    def repl(match: re.Match[str]) -> str:
        column = match.group(1)
        not_part = bool(match.group(2))
        values = match.group(3)
        items = [item.strip() for item in values.split(",")]
        list_repr = "[" + ", ".join(items) + "]"
        operator = "not in" if not_part else "in"
        return f"{column} {operator} {list_repr}"

    return pattern.sub(repl, expr)


def _normalize_logical_keywords(expr: str) -> str:
    """Lower-case SQL logical keywords so pandas query strings behave."""
    expr = re.sub(r"(?i)\bAND\b", " and ", expr)
    expr = re.sub(r"(?i)\bOR\b", " or ", expr)
    expr = re.sub(r"(?i)\bNOT\b", " not ", expr)
    return expr


def _normalize_where_expression(expr: str) -> str:
    """Perform all literal and operator conversions required for pandas query strings."""
    expr = _convert_sas_date_literals(expr)
    expr = _expand_between_conditions(expr)
    expr = _convert_in_clauses(expr)
    expr = re.sub(r"<>", "!=", expr)
    expr = re.sub(r"(?<![<>=!])=(?!=)", "==", expr)
    expr = _normalize_logical_keywords(expr)
    expr = re.sub(r"\s+", " ", expr).strip()
    return _strip_table_prefixes(expr)


def _split_alias(select_item: str) -> Tuple[str, Optional[str]]:
    """Split a select clause entry into the expression and optional alias."""
    item = select_item.strip()
    parts = re.split(r"\s+AS\s+", item, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    tokens = item.rsplit(None, 1)
    if len(tokens) == 2 and re.match(r"[A-Za-z_]\w*$", tokens[1]):
        return tokens[0].strip(), tokens[1].strip()
    return item, None


def _parse_aggregate(expr: str) -> Optional[Tuple[str, str, bool]]:
    """Return aggregate metadata (function, argument, distinct flag) if expression is an aggregate."""
    match = re.match(r"(?i)^\s*([a-z]+)\s*\(\s*(distinct\s+)?([^\)]+)\s*\)\s*$", expr)
    if not match:
        return None
    fn = match.group(1).lower()
    is_distinct = bool(match.group(2))
    arg = match.group(3).strip()
    return fn, arg, is_distinct


def _format_list(values: List[str]) -> str:
    """Render a python list literal from a list of strings."""
    return "[" + ", ".join(repr(v) for v in values) + "]"


def _canonicalize_expression(expr: str) -> str:
    """Create a canonical key for an expression so aliases match regardless of whitespace."""
    expr = _convert_sas_date_literals(expr)
    expr = _expand_between_conditions(expr)
    expr = _convert_in_clauses(expr)
    stripped = _strip_table_prefixes(expr)
    return re.sub(r"\s+", "", stripped).lower()


def _translate_having_expression(
    expr: str,
    aggregate_alias_map: Dict[str, str],
) -> str:
    """Convert a HAVING clause into a pandas query using aggregate aliases."""
    expr = _convert_sas_date_literals(expr)
    expr = _expand_between_conditions(expr)
    expr = _convert_in_clauses(expr)

    func_pattern = re.compile(r"[A-Za-z_][\w]*\s*\([^()]*\)")

    def replace_func(match: re.Match[str]) -> str:
        canonical = _canonicalize_expression(match.group(0))
        alias = aggregate_alias_map.get(canonical)
        return alias if alias else match.group(0)

    expr = func_pattern.sub(replace_func, expr)
    expr = _normalize_logical_keywords(expr)
    expr = expr.replace("<>", "!=")
    expr = re.sub(r"(?<![<>=!])=(?!=)", "==", expr)
    expr = re.sub(r"\s+", " ", expr).strip()
    return _strip_table_prefixes(expr)


def to_pandas_code(q: SQLQuery, table_map: Dict[str, str]) -> str:
    """Generate pandas code that mirrors the logic described by the SQLQuery."""
    lines: List[str] = []
    if q.from_table is None:
        raise ValueError("FROM table is required to generate pandas code.")
    base_key = q.from_table
    base = table_map.get(base_key, base_key)
    if not base:
        raise ValueError("FROM table is required to generate pandas code.")

    column_alias_map: Dict[str, str] = {}

    lines.append("# Generated by SQLift PoC")
    lines.append(f"df = {base}.copy()")

    for j in q.joins:
        right = table_map.get(j["table"], j["table"])
        left_key = j["left"].split(".")[-1]
        right_key = j["right"].split(".")[-1]
        if left_key == right_key:
            lines.append(f"df = df.merge({right}, on='{left_key}', how='inner')")
        else:
            lines.append(
                f"df = df.merge({right}, left_on='{left_key}', right_on='{right_key}', how='inner')"
            )

    if q.where:
        expr = _normalize_where_expression(q.where)
        lines.append(f"df = df.query({expr!r})")

    if q.group_by:
        gb_cols = [c.split(".")[-1] for c in q.group_by if c.strip()]
        agg_map: "OrderedDict[str, Tuple[str, str]]" = OrderedDict()
        rename_map: Dict[str, str] = {}
        projected_names: List[str] = []
        aggregate_alias_map: Dict[str, str] = {}

        for sel in q.select:
            expr, alias = _split_alias(sel)
            agg_info = _parse_aggregate(expr)
            if agg_info:
                fn, arg, is_distinct = agg_info
                cleaned_arg = _strip_table_prefixes(arg).strip()
                column_name = cleaned_arg.split(".")[-1]
                if column_name in {"*", "1"}:
                    if not gb_cols:
                        raise ValueError(
                            "COUNT(*) requires at least one GROUP BY column."
                        )
                    source_col = gb_cols[0]
                else:
                    source_col = column_name

                pandas_fn = {"avg": "mean"}.get(fn, fn)
                if fn == "count":
                    if is_distinct:
                        pandas_fn = "nunique"
                    elif column_name in {"*", "1"}:
                        pandas_fn = "size"
                    else:
                        pandas_fn = "count"
                elif is_distinct:
                    pandas_fn = "nunique"

                output_name = (alias or f"{fn}_{source_col}").replace("*", "all")
                agg_map[output_name] = (source_col, pandas_fn)
                projected_names.append(output_name)
                column_alias_map[output_name] = output_name
                column_alias_map[_canonicalize_expression(expr)] = output_name
                aggregate_alias_map[_canonicalize_expression(expr)] = output_name
            else:
                column_name = _strip_table_prefixes(expr).split(".")[-1]
                output_name = alias or column_name
                if alias and alias != column_name:
                    rename_map[column_name] = output_name
                projected_names.append(output_name)
                column_alias_map[column_name] = output_name
                column_alias_map[output_name] = output_name
                column_alias_map[_canonicalize_expression(expr)] = output_name

        if agg_map:
            size_shortcut = False
            size_alias = None
            if len(agg_map) == 1:
                only_alias, (only_col, only_fn) = next(iter(agg_map.items()))
                if only_fn == "size":
                    size_shortcut = True
                    size_alias = only_alias

            if size_shortcut:
                if len(gb_cols) == 1:
                    gb_repr = repr(gb_cols[0])
                else:
                    gb_repr = _format_list(gb_cols)
                rename_clause = "{'size': " + repr(size_alias) + "}"
                lines.append(
                    "df = df.groupby("
                    f"{gb_repr}, as_index=False).size().rename("
                    f"columns={rename_clause})"
                )
                if size_alias is None:
                    raise ValueError("COUNT(*) aggregate requires an alias.")
                actual_order = [
                    rename_map[col] if col in rename_map else col for col in gb_cols
                ] + [size_alias]
            else:
                gb_repr = _format_list(gb_cols)
                named_entries = ", ".join(
                    f"{repr(alias)}: ({repr(col)}, {repr(fn)})"
                    for alias, (col, fn) in agg_map.items()
                )
                lines.append(
                    f"df = df.groupby({gb_repr}, as_index=False).agg({{{named_entries}}})"
                )
                actual_order = [
                    rename_map[col] if col in rename_map else col for col in gb_cols
                ] + list(agg_map.keys())
        else:
            subset_repr = _format_list(gb_cols)
            lines.append(f"df = df.drop_duplicates(subset={subset_repr})")
            actual_order = [
                rename_map[col] if col in rename_map else col for col in gb_cols
            ]

        if rename_map:
            rename_repr = (
                "{"
                + ", ".join(
                    f"{repr(src)}: {repr(dst)}" for src, dst in rename_map.items()
                )
                + "}"
            )
            lines.append(f"df = df.rename(columns={rename_repr})")
            for src, dst in rename_map.items():
                column_alias_map[src] = dst
                column_alias_map[_canonicalize_expression(src)] = dst
                column_alias_map[dst] = dst
            actual_order = [
                rename_map[col] if col in rename_map else col for col in actual_order
            ]

        if q.having and aggregate_alias_map:
            having_expr = _translate_having_expression(q.having, aggregate_alias_map)
            lines.append(f"df = df.query({having_expr!r})")

        select_needed = False
        if not agg_map and projected_names:
            select_needed = True
        elif projected_names and projected_names != actual_order:
            select_needed = True
        if select_needed:
            cols_repr = _format_list(projected_names)
            lines.append(f"df = df[{cols_repr}]")

    else:
        if q.select and q.select[0] != "*":
            source_cols: List[str] = []
            final_cols: List[str] = []
            rename_map_simple: Dict[str, str] = {}

            for sel in q.select:
                expr, alias = _split_alias(sel)
                column_name = _strip_table_prefixes(expr).split(".")[-1]
                final_name = alias or column_name
                source_cols.append(column_name)
                final_cols.append(final_name)
                if alias and alias != column_name:
                    rename_map_simple[column_name] = final_name
                column_alias_map[column_name] = final_name
                column_alias_map[final_name] = final_name
                column_alias_map[_canonicalize_expression(expr)] = final_name

            src_repr = _format_list(source_cols)
            lines.append(f"df = df[{src_repr}]")
            if rename_map_simple:
                rename_repr = (
                    "{"
                    + ", ".join(
                        f"{repr(src)}: {repr(dst)}"
                        for src, dst in rename_map_simple.items()
                    )
                    + "}"
                )
                lines.append(f"df = df.rename(columns={rename_repr})")
                for src, dst in rename_map_simple.items():
                    column_alias_map[src] = dst
                    column_alias_map[_canonicalize_expression(src)] = dst
                    column_alias_map[dst] = dst
            if final_cols != source_cols:
                cols_repr = _format_list(final_cols)
                lines.append(f"df = df[{cols_repr}]")
        else:
            for col in q.select:
                clean = _strip_table_prefixes(col).split(".")[-1]
                column_alias_map[clean] = clean

    if q.order_by:
        by: List[str] = []
        asc: List[bool] = []
        for term in q.order_by:
            parts = term.split()
            raw_expr = parts[0]
            canonical = _canonicalize_expression(raw_expr)
            raw = raw_expr.split(".")[-1]
            sort_key = column_alias_map.get(canonical)
            if sort_key is None:
                sort_key = column_alias_map.get(raw, raw)
            by.append(sort_key)
            asc.append(False if len(parts) > 1 and parts[1].lower() == "desc" else True)
        if len(by) == 1:
            lines.append(f"df = df.sort_values(by={repr(by[0])}, ascending={asc[0]})")
        else:
            lines.append(f"df = df.sort_values(by={_format_list(by)}, ascending={asc})")

    if q.limit:
        lines.append(f"df = df.head({q.limit})")

    return "\n".join(lines)


def main():
    """Command-line entry point for generating pandas code from SQL."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sql", required=True, help="Path to SQL file")
    ap.add_argument(
        "--table",
        action="append",
        default=[],
        help="Mapping like table=df_alias (repeatable)",
    )
    ap.add_argument(
        "--out",
        default="generated_output.py",
        help="File to write generated pandas code",
    )
    args = ap.parse_args()

    with open(args.sql, "r", encoding="utf-8") as f:
        sql = f.read()

    q = parse_sql(sql)
    table_map: Dict[str, str] = {}
    for t in args.table:
        if "=" in t:
            k, v = t.split("=", 1)
            table_map[k.strip()] = v.strip()

    code = to_pandas_code(q, table_map)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(code + "\n")
    print(f"Generated -> {args.out}")
    print(code)


if __name__ == "__main__":
    main()
