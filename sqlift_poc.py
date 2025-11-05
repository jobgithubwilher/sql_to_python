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
    def __init__(self):
        """
        Define a SQL Query as a class and identify each part as an attribute.

        Attributes:
            select (List[str]):
                List of columns or expressions to retrieve.
                Example: ["user_id", "COUNT(order_id) AS total_orders"]

            from_table (Optional[str]):
                The main table to select data from.
                Example: "users"

            joins (List[Dict[str, Any]]):
                List of JOIN clauses, where each join is represented as a dictionary
                describing the type and condition.
                Example: [
                    {"type": "INNER", "table": "orders", "on": "users.user_id = orders.user_id"}
                ]

            where (Optional[str]):
                The WHERE condition used to filter rows.
                Example: "users.country = 'Germany' AND orders.amount > 100"

            having (Optional[str]):
                The HAVING condition applied after GROUP BY.
                Example: "COUNT(order_id) > 5"

            group_by (List[str]):
                Columns or expressions used for grouping results.
                Example: ["user_id", "country"]

            order_by (List[str]):
                Columns or expressions used to sort the result set.
                Example: ["total_orders DESC", "user_id ASC"]

            limit (Optional[int]):
                Maximum number of rows to return.
                Example: 100
        """
        self.select: List[str] = []
        self.from_table: Optional[str] = None
        self.joins: List[Dict[str, Any]] = []
        self.where: Optional[str] = None
        self.having: Optional[str] = None
        self.group_by: List[str] = []
        self.order_by: List[str] = []
        self.limit: Optional[int] = None


def _normalize_whitespace(s: str) -> str:
    """
    Collapse whitespace so regex parsing can target single-line SQL fragments.

    This function removes leading/trailing spaces, replaces newlines with spaces,
    and collapses consecutive whitespace characters into a single space.

    Example:
        >>> _normalize_whitespace("SELECT *  \n   FROM   users \nWHERE id = 1;")
        'SELECT * FROM users WHERE id = 1;'
    """
    return " ".join(s.strip().replace("\n", " ").split())


def parse_sql(sql: str) -> SQLQuery:
    """
    Parse a limited SQL SELECT statement into an intermediate SQLQuery object.

    This function performs basic parsing of SQL queries to extract key clauses
    (SELECT, FROM, JOIN, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT) and store them
    in an `SQLQuery` object for programmatic manipulation or analysis.

    Args:
        sql (str): SQL SELECT statement to parse.

    Returns:
        SQLQuery: Structured representation of the SQL query.
    """

    # Normalize all whitespace (replace tabs, newlines, and multiple spaces with a single space)
    sql = _normalize_whitespace(sql)

    # Create a new SQLQuery instance to hold parsed components
    q = SQLQuery()

    # -----------------------------
    # Parse SELECT clause
    # -----------------------------
    # Match everything between "SELECT" and "FROM"
    m = re.search(r"select\s+(.*?)\s+from\s+", sql, re.IGNORECASE)
    if m:
        # Split by comma and strip whitespace from each selected column/expression
        q.select = [x.strip() for x in m.group(1).split(",")]

    # -----------------------------
    # Parse FROM clause
    # -----------------------------
    # Capture the table name following "FROM"
    m = re.search(r"from\s+([a-zA-Z0-9_\.]+)", sql, re.IGNORECASE)
    if m:
        q.from_table = m.group(1)

    # -----------------------------
    # Parse JOIN clauses
    # -----------------------------
    # Find all JOIN patterns like "JOIN table ON left = right"
    for j in re.finditer(
        r"join\s+([a-zA-Z0-9_\.]+)"  # Joined table name
        r"(?:\s+(?:as\s+)?[a-zA-Z0-9_\.]+)?"  # Optional alias (ignored)
        r"\s+on\s+([^\s]+)\s*=\s*([^\s]+)",  # ON condition: left = right
        sql,
        re.IGNORECASE,
    ):
        # Append each join as a dictionary to q.joins
        q.joins.append(
            {
                "table": j.group(1),  # joined table
                "left": j.group(2),  # left column in join condition
                "right": j.group(3),  # right column in join condition
            }
        )

    # -----------------------------
    # Parse WHERE clause
    # -----------------------------
    # Capture everything after "WHERE" until the next clause keyword or end of string
    m = re.search(
        r"where\s+(.*?)\s*(group by|having|order by|limit|$)",
        sql,
        re.IGNORECASE,
    )
    if m:
        q.where = m.group(1)

    # -----------------------------
    # Parse GROUP BY clause
    # -----------------------------
    # Capture columns in the GROUP BY clause, stopping at next keyword
    m = re.search(
        r"group by\s+(.*?)\s*(having|order by|limit|$)",
        sql,
        re.IGNORECASE,
    )
    if m:
        # Split multiple columns by comma and clean whitespace
        q.group_by = [x.strip() for x in m.group(1).split(",")]

    # -----------------------------
    # Parse HAVING clause
    # -----------------------------
    # Capture HAVING condition
    m = re.search(r"having\s+(.*?)\s*(order by|limit|$)", sql, re.IGNORECASE)
    if m:
        q.having = m.group(1).strip()

    # -----------------------------
    # Parse ORDER BY clause
    # -----------------------------
    # Capture ORDER BY expressions
    m = re.search(r"order by\s+(.*?)\s*(limit|$)", sql, re.IGNORECASE)
    if m:
        entries = []
        # Split by commas in case of multiple order expressions
        for part in m.group(1).split(","):
            # Strip whitespace and trailing semicolons
            cleaned = part.strip().rstrip(";")
            if cleaned:
                entries.append(cleaned)
        q.order_by = entries

    # -----------------------------
    # Parse LIMIT clause
    # -----------------------------
    # Capture numeric value after LIMIT
    m = re.search(r"limit\s+(\d+)", sql, re.IGNORECASE)
    if m:
        # Convert string to integer
        q.limit = int(m.group(1))

    # -----------------------------
    # Return final structured SQLQuery object
    # -----------------------------
    return q


def _strip_table_prefixes(expr: str) -> str:
    """
    Remove table or alias prefixes (e.g., 't1.column') from a SQL expression,
    without modifying text enclosed in single or double quotes.

    This function scans a SQL expression character by character and removes
    table name prefixes before column references, while preserving quoted
    strings and identifiers.

    Example:
        Input:  "t1.name = 'table.column' AND t2.age > 18"
        Output: "name = 'table.column' AND age > 18"

    Args:
        expr (str): A SQL expression string (e.g., part of a WHERE, SELECT, or ON clause).

    Returns:
        str: The expression with table/alias prefixes removed.
    """

    # The final processed characters will be collected here
    result: List[str] = []

    # Index pointer to walk through each character in the expression
    i = 0

    # Flags to track whether we are currently inside single or double quotes
    in_single = False
    in_double = False

    # Iterate through each character in the string
    while i < len(expr):
        ch = expr[i]  # current character

        # ----------------------------------------------------
        # Handle quoted strings (single or double quotes)
        # ----------------------------------------------------
        if ch == "'" and not in_double:
            # Toggle single quote mode (enter or exit)
            in_single = not in_single
            result.append(ch)
            i += 1
            continue

        if ch == '"' and not in_single:
            # Toggle double quote mode (enter or exit)
            in_double = not in_double
            result.append(ch)
            i += 1
            continue

        # ----------------------------------------------------
        # Handle identifiers (table/column names)
        # Only process when NOT inside quotes.
        # ----------------------------------------------------
        if not in_single and not in_double and (ch.isalpha() or ch == "_"):
            j = i

            # Move forward while characters are part of a valid identifier
            # (letters, numbers, underscores, or dots)
            while j < len(expr) and (expr[j].isalnum() or expr[j] in "._"):
                j += 1

            # Extract the token (could be "t1.column" or just "column")
            token = expr[i:j]

            # If the token contains a dot, remove everything before it
            # e.g., "t1.column" → "column"
            if "." in token:
                result.append(token.split(".")[-1])
            else:
                # Otherwise, keep it as is
                result.append(token)

            # Skip ahead to the next unprocessed character
            i = j

        else:
            # For non-identifier characters (spaces, operators, punctuation)
            # just copy them as-is.
            result.append(ch)
            i += 1

    # Join the list of characters back into a single string
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
    """
    Convert SAS-style date literals (e.g., `'01JAN2024'd`) into ISO 8601 date strings (e.g., `'2024-01-01'`).

    SAS date literals are written as `'DDMONYYYY'd`, where:
        - `DD`  = day (1–31)
        - `MON` = three-letter month abbreviation (e.g., JAN, FEB, MAR)
        - `YYYY` = four-digit year

    This function identifies all such literals in a string using regex and replaces them
    with equivalent ISO-formatted strings `'YYYY-MM-DD'`.

    Example:
        Input:  "WHERE signup_date >= '01JAN2024'd AND expiry_date < '15FEB2024'd"
        Output: "WHERE signup_date >= '2024-01-01' AND expiry_date < '2024-02-15'"

    Args:
        expr (str): A SQL or SAS-like expression possibly containing SAS date literals.

    Returns:
        str: The expression with all SAS-style date literals converted to ISO strings.

    """

    # ------------------------------------------------------------
    # Inner replacement function called for each regex match
    # ------------------------------------------------------------
    def repl(match: re.Match[str]) -> str:
        # Extract the day, month abbreviation, and year from the matched pattern
        day, mon, year = match.group(1), match.group(2).upper(), match.group(3)

        # Convert month abbreviation (e.g., 'JAN') to numeric month (e.g., '01')
        # `_MONTH_MAP` should map 'JAN' → '01', 'FEB' → '02', etc.
        # Default to '01' if not found
        iso = f"{year}-{_MONTH_MAP.get(mon, '01')}-{int(day):02d}"

        # Return the new ISO date wrapped in single quotes (e.g., '2024-01-01')
        return f"'{iso}'"

    return re.sub(r"'(\d{1,2})([A-Za-z]{3})(\d{4})'d", repl, expr)


def _expand_between_conditions(expr: str) -> str:
    """
    Rewrite SQL BETWEEN expressions into equivalent explicit comparisons.

    This function identifies expressions of the form:
        <column> BETWEEN <lower> AND <upper>
    and rewrites them into:
        (<column> >= <lower>) AND (<column> <= <upper>)

    Example:
        Input:  "age BETWEEN 18 AND 65"
        Output: "(age >= 18) AND (age <= 65)"

    It supports identifiers containing underscores and dots (e.g., `table.column`),
    and is case-insensitive to the "BETWEEN" and "AND" keywords.

    Args:
        expr (str): SQL expression possibly containing BETWEEN ... AND ... clauses.

    Returns:
        str: Expression with all BETWEEN clauses expanded into explicit comparisons.
    """

    pattern = re.compile(r"(?i)([A-Za-z_][\w\.]*)\s+BETWEEN\s+([^ ]+)\s+AND\s+([^ ]+)")

    # ------------------------------------------------------------
    # Inner function to replace each BETWEEN match
    # ------------------------------------------------------------
    def repl(match: re.Match[str]) -> str:
        # Extract column name, lower bound, and upper bound
        col = match.group(1)
        lower = match.group(2)
        upper = match.group(3).rstrip(";")  # remove trailing semicolon if any

        # Return expanded explicit condition
        return f"({col} >= {lower}) and ({col} <= {upper})"

    # ------------------------------------------------------------
    # Replace all BETWEEN occurrences in the input expression
    # ------------------------------------------------------------
    return pattern.sub(repl, expr)


def _convert_in_clauses(expr: str) -> str:
    """
    Translate SQL IN/NOT IN predicates into pandas-friendly membership checks.

    This helper rewrites clauses such as ``country IN ('US', 'MX')`` or
    ``status NOT IN (1, 2, 3)`` into Python-style list membership expressions
    so they can be executed inside `DataFrame.query`. Items are left untouched
    so downstream consumers retain control over quoting.

    Example:
        >>> _convert_in_clauses("country IN ('US', 'MX')")
        "country in ['US', 'MX']"

    Args:
        expr (str): SQL fragment that may contain IN or NOT IN predicates.

    Returns:
        str: Expression with membership tests rewritten as ``column in [...]`` or
            ``column not in [...]``.
    """
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
    """
    Normalize logical operators so pandas query strings use Python keywords.

    Pandas expects boolean operators written as lowercase ``and``, ``or``, and
    ``not`` with surrounding spaces. This function replaces SQL-style uppercase
    keywords to avoid syntax errors when the expression is evaluated.

    Example:
        >>> _normalize_logical_keywords("A AND (B OR NOT C)")
        'A and (B or not C)'

    Args:
        expr (str): Boolean expression from a WHERE or HAVING clause.

    Returns:
        str: Expression with logical keywords lowercased and space padded.
    """
    expr = re.sub(r"(?i)\bAND\b", " and ", expr)
    expr = re.sub(r"(?i)\bOR\b", " or ", expr)
    expr = re.sub(r"(?i)\bNOT\b", " not ", expr)
    return expr


def _normalize_where_expression(expr: str) -> str:
    """
    Apply the full suite of literal/operator rewrites required for pandas queries.

    The WHERE clause in SQL often uses syntax that `DataFrame.query` does not
    understand directly. This routine orchestrates several helper transforms:
    converting SAS date literals, expanding BETWEEN ranges, transforming IN
    predicates, normalizing equality/inequality operators, and stripping table
    prefixes so the resulting expression can be executed against a DataFrame.

    Example:
        >>> _normalize_where_expression("users.age BETWEEN 18 AND 30 AND country <> 'US'")
        "(age >= 18) and (age <= 30) and country != 'US'"

    Args:
        expr (str): Raw WHERE clause expression copied from the SQL statement.

    Returns:
        str: Cleaned expression compatible with pandas query semantics.
    """
    expr = _convert_sas_date_literals(expr)
    expr = _expand_between_conditions(expr)
    expr = _convert_in_clauses(expr)
    expr = re.sub(r"<>", "!=", expr)
    expr = re.sub(r"(?<![<>=!])=(?!=)", "==", expr)
    expr = _normalize_logical_keywords(expr)
    expr = re.sub(r"\s+", " ", expr).strip()
    return _strip_table_prefixes(expr)


def _split_alias(select_item: str) -> Tuple[str, Optional[str]]:
    """
    Split a SELECT clause entry into the core expression and optional alias.

    The SQL parser supports both ``expr AS alias`` and the positional
    ``expr alias`` shorthand that some dialects allow. This helper inspects the
    token and returns the raw expression plus the alias (or ``None`` when missing).

    Example:
        >>> _split_alias("SUM(total) AS revenue")
        ('SUM(total)', 'revenue')

    Args:
        select_item (str): Single comma-separated entry taken from the SELECT list.

    Returns:
        Tuple[str, Optional[str]]: ``(expression, alias)``, where alias is ``None``
            if the input did not specify one.
    """
    item = select_item.strip()
    parts = re.split(r"\s+AS\s+", item, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    tokens = item.rsplit(None, 1)
    if len(tokens) == 2 and re.match(r"[A-Za-z_]\w*$", tokens[1]):
        return tokens[0].strip(), tokens[1].strip()
    return item, None


def _parse_aggregate(expr: str) -> Optional[Tuple[str, str, bool]]:
    """
    Extract aggregate metadata from a SELECT expression, if present.

    The parser needs to know which columns participate in aggregations so it can
    build the appropriate pandas `groupby` call. This helper recognises common
    aggregate patterns such as ``SUM(amount)`` or ``COUNT(DISTINCT user_id)`` and
    returns the function name, its argument, and whether the DISTINCT keyword was
    supplied.

    Example:
        >>> _parse_aggregate("COUNT(DISTINCT orders.id)")
        ('count', 'orders.id', True)

    Args:
        expr (str): SELECT list expression to inspect.

    Returns:
        Optional[Tuple[str, str, bool]]: ``(function, argument, is_distinct)`` when
            the expression is an aggregate, otherwise ``None``.
    """
    match = re.match(r"(?i)^\s*([a-z]+)\s*\(\s*(distinct\s+)?([^\)]+)\s*\)\s*$", expr)
    if not match:
        return None
    fn = match.group(1).lower()
    is_distinct = bool(match.group(2))
    arg = match.group(3).strip()
    return fn, arg, is_distinct


def _format_list(values: List[str]) -> str:
    """
    Render a Python list literal string from a sequence of column names.

    This utility centralises how we generate list representations in the emitted
    pandas code so the formatting stays consistent (e.g., quoting and commas).

    Example:
        >>> _format_list(['user_id', 'country'])
        "['user_id', 'country']"

    Args:
        values (List[str]): Column names or other string tokens to render.

    Returns:
        str: String containing a valid Python list literal.
    """
    return "[" + ", ".join(repr(v) for v in values) + "]"


def _canonicalize_expression(expr: str) -> str:
    """
    Produce a canonical key for SQL expressions so alias lookups remain stable.

    By normalizing literals, expanding composite operators, stripping table
    prefixes, and collapsing whitespace, we can compare expressions such as
    ``SUM(t.amount)`` and ``sum(amount)`` reliably when mapping SELECT aliases.

    Example:
        >>> _canonicalize_expression("SUM ( t.amount )")
        'sum(amount)'

    Args:
        expr (str): Raw SQL expression (often a SELECT entry or ORDER BY term).

    Returns:
        str: Lower-cased canonical representation suitable for dictionary keys.
    """
    expr = _convert_sas_date_literals(expr)
    expr = _expand_between_conditions(expr)
    expr = _convert_in_clauses(expr)
    stripped = _strip_table_prefixes(expr)
    return re.sub(r"\s+", "", stripped).lower()


def _translate_having_expression(
    expr: str,
    aggregate_alias_map: Dict[str, str],
) -> str:
    """
    Convert a HAVING clause into a pandas query string that uses aggregate aliases.

    After aggregation we often rename outputs (e.g., ``COUNT(*)`` → ``order_count``).
    This function replaces aggregate function calls in the HAVING expression with
    those aliases and then runs the usual normalization pipeline so the result
    can be evaluated with `DataFrame.query`.

    Example:
        >>> _translate_having_expression("COUNT(*) > 5", {'count(*)': 'order_count'})
        'order_count > 5'

    Args:
        expr (str): Original HAVING clause expression pulled from the SQL.
        aggregate_alias_map (Dict[str, str]): Mapping of canonical aggregate
            expressions to their projected column names.

    Returns:
        str: HAVING expression rewritten to reference aggregated pandas columns.
    """
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
    """
    Generate executable pandas code that mirrors the parsed SQL query.

    The converter walks each clause (FROM, JOIN, WHERE, GROUP BY, HAVING,
    ORDER BY, LIMIT) and emits the corresponding DataFrame operations. The
    resulting string is intended to be written to a Python file or executed
    within a Dataiku recipe without additional editing.

    Args:
        q (SQLQuery): Structured representation of the input SQL statement.
        table_map (Dict[str, str]): Mapping from SQL table names to pandas
            variables already present in the notebook or recipe.

    Returns:
        str: Multi-line Python source code that reproduces the SQL semantics.

    Raises:
        ValueError: If required metadata (such as the FROM table) is missing.
    """
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
    """
    Command-line entry point for transforming SQL files into pandas code snippets.

    The CLI accepts a path to a SQL file plus optional table mappings that bind
    SQL identifiers to in-memory DataFrames. It writes the generated code to the
    requested output file and echoes it to stdout for quick inspection.

    Args:
        None directly; arguments are parsed from ``sys.argv``.

    Side Effects:
        Reads the SQL file from disk, writes the generated Python file, and
        prints a short status message.
    """
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
