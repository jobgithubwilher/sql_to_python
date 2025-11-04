#!/usr/bin/env python3
"""Streamlit front-end for SQLift."""

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import streamlit as st

from sqlift_poc import parse_sql, to_pandas_code


def _load_default_sql() -> str:
    sample_path = Path("example_input.sql")
    if sample_path.exists():
        try:
            return sample_path.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return (
        "SELECT c.customer_id\n"
        "FROM orders o\n"
        "JOIN customers c ON o.customer_id = c.customer_id\n"
        "WHERE o.status = 'PAID'\n"
        "ORDER BY total_amount DESC\n"
        "LIMIT 100;"
    )


def _parse_table_map(raw: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for line in raw.splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if "=" not in text:
            raise ValueError(
                ("Invalid mapping line: " f"{text!r}. Expected format table=df_alias.")
            )
        table, alias = text.split("=", 1)
        table = table.strip()
        alias = alias.strip()
        if not table or not alias:
            raise ValueError(
                (
                    "Invalid mapping line: "
                    f"{text!r}. Table and alias must be non-empty."
                )
            )
        mapping[table] = alias
    return mapping


def _extract_sql_from_proc(proc_sql: str) -> Tuple[str, Optional[int]]:
    lines = []
    limit: Optional[int] = None
    for raw in proc_sql.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("PROC SQL"):
            match = re.search(r"OUTOBS\s*=\s*(\d+)", line, re.IGNORECASE)
            if match:
                limit = int(match.group(1))
            continue
        if line.upper() == "QUIT;" or line.upper() == "QUIT":
            continue
        if line.upper().startswith("CREATE TABLE"):
            # PROC SQL often has CREATE TABLE foo AS SELECT ...
            # we just skip the CREATE TABLE ... AS portion if present.
            parts = line.split("SELECT", 1)
            if len(parts) == 2:
                line = "SELECT" + parts[1]
        lines.append(line)
    return "\n".join(lines).strip(), limit


def main() -> None:
    st.set_page_config(page_title="SQLift", page_icon="🧯")
    st.title("SQLift — SQL → pandas code generator")
    st.write(
        "Paste a SQL query and provide table-to-DataFrame mappings. "
        "SQLift will generate pandas code you can drop into a Dataiku Python "
        "recipe."
    )

    default_sql = _load_default_sql()
    col_standard, col_proc = st.columns(2)
    with col_standard:
        sql_input = st.text_area(
            "Standard SQL query",
            value=default_sql,
            height=280,
            help="Paste ANSI SQL you want to translate.",
        )
        generate_standard = st.button("Generate from Standard SQL")

    with col_proc:
        proc_sql_input = st.text_area(
            "PROC SQL snippet",
            value="PROC SQL;\nSELECT * FROM work.orders;\nQUIT;",
            height=280,
            help=(
                "Optional SAS PROC SQL block. We'll extract the SELECT "
                "statement if present."
            ),
        )
        generate_proc = st.button("Generate from PROC SQL")

    chosen_sql = ""
    source_label = ""
    override_limit: Optional[int] = None

    if generate_standard:
        chosen_sql = sql_input.strip()
        source_label = "Standard SQL"
    elif generate_proc:
        source_label = "PROC SQL"
        chosen_sql, override_limit = _extract_sql_from_proc(proc_sql_input.strip())
        if chosen_sql:
            st.info("Extracted SELECT statement from PROC SQL snippet.")

    if generate_standard or generate_proc:
        if not chosen_sql:
            st.warning(f"No SQL detected in the {source_label or 'selected'} input.")
            return
        try:
            table_map: Dict[str, str] = {}
            query = parse_sql(chosen_sql)
            if override_limit is not None and (
                query.limit is None or override_limit < query.limit
            ):
                query.limit = override_limit
            code = to_pandas_code(query, table_map)
        except Exception as exc:  # pragma: no cover - surfaced to UI
            st.error(f"Failed to generate code: {exc}")
        else:
            st.success(f"Pandas code generated from {source_label}.")
            st.code(code, language="python")

            with st.expander("Parsed query (debug)"):
                st.json(
                    {
                        "select": query.select,
                        "from": query.from_table,
                        "joins": query.joins,
                        "where": query.where,
                        "group_by": query.group_by,
                        "order_by": query.order_by,
                        "limit": query.limit,
                    }
                )


if __name__ == "__main__":
    main()
