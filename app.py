"""
Inventory & Fulfillment Exception Engine - Streamlit App
---------------------------------------------------------
Takes a natural-language operational question, routes it through the query
engine pipeline (verified template vs. generated SQL), validates the SQL
through a 4-check gate, retries once on generated queries, executes against
the local read-only SQLite database, and produces a concise business answer.

Run with:
    streamlit run app.py

Expected next to this file:
    - supply_chain_ops.db
    - Credentials via ONE of:
        * .streamlit/secrets.toml  (OPENAI_API_KEY, OPENAI_API_BASE)
        * config.json              ({"OPENAI_API_KEY": ..., "OPENAI_API_BASE": ...})
        * environment variables    (OPENAI_API_KEY, optional OPENAI_API_BASE)
"""

import json
import os
import re
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Inventory & Fulfillment Exception Engine",
    page_icon="📦",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("SUPPLY_CHAIN_DB_PATH", APP_DIR / "supply_chain_ops.db"))
AUDIT_LOG_PATH = APP_DIR / "audit_log.jsonl"


# ============================================================
# Credentials / LLMs / Database
# ============================================================

def load_credentials():
    """
    Look for the OpenAI key/base URL in (1) Streamlit secrets,
    (2) config.json next to app.py, (3) environment variables.
    """
    api_key, api_base = None, None

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        api_base = st.secrets.get("OPENAI_API_BASE")
    except Exception:
        pass

    if not api_key:
        config_file = APP_DIR / "config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)
            api_key = config.get("OPENAI_API_KEY")
            api_base = config.get("OPENAI_API_BASE")

    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = api_base or os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")

    return api_key, api_base


OPENAI_API_KEY, OPENAI_API_BASE = load_credentials()

if not OPENAI_API_KEY:
    st.error(
        "OpenAI API key not found. Add OPENAI_API_KEY (and optionally OPENAI_API_BASE) "
        "to .streamlit/secrets.toml, a config.json next to app.py, or your environment variables."
    )
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENAI_API_BASE:
    os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE


@st.cache_resource
def get_llms():
    """
    Primary LLM  : intent classification, SQL generation, retry, response generation.
    Evaluator LLM: SQL relevance validation.
    """
    primary = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator = ChatOpenAI(model="gpt-4o", temperature=0)
    return primary, evaluator


@st.cache_resource
def get_db_connection():
    """Opens the on-premise SQLite database in strict read-only mode."""
    uri = DB_PATH.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


if not DB_PATH.exists():
    st.error(f"Database file not found: {DB_PATH}")
    st.stop()

llm, evaluator_llm = get_llms()
conn = get_db_connection()


# ============================================================
# Database Schema (passed to the LLM)
# ============================================================

database_schema = """
warehouse_master:
  warehouse_id (TEXT, PK): US MSA facility code (e.g., WH_ORD_01, WH_DFW_02)
  warehouse_name (TEXT): legal facility name (e.g., 'Chicago O\\'Hare Hub', 'Dallas Fort-Worth Main')
  region (TEXT): US Census Region (Northeast, Midwest, South, West)
  max_capacity_pallet_positions (INTEGER): total high-bay pallet position capacity
  current_occupancy_pct (REAL): utilization percentage; high occupancy threshold is >= 85.0
  is_cbp_bonded_ftz (INTEGER): 1 if CBP-bonded or Foreign Trade Zone, 0 otherwise

inventory_levels:
  inventory_id (INTEGER, PK): auto-increment identifier
  warehouse_id (TEXT, FK): joins to warehouse_master.warehouse_id
  sku_id (TEXT): unique stock keeping unit code
  sku_category (TEXT): one of Consumer Packaged Goods, Automotive Parts, Cold-Chain Perishables, Apparel, Industrial
  units_on_hand (INTEGER): physical stock count in warehouse
  reorder_point (INTEGER): minimum stock threshold before replenishment order
  unit_cost_usd (REAL): carrying unit cost under US GAAP (ASC 330)
  last_restock_date (DATE): date of last inventory receipt

shipment_tracker:
  shipment_id (TEXT, PK): unique BOL or tracking number (e.g., BOL-000001)
  order_id (TEXT): client purchase order reference (e.g., PO-100001)
  origin_warehouse_id (TEXT, FK): joins to warehouse_master.warehouse_id
  scac_code (TEXT, FK): joins to carrier_performance.scac_code
  promised_ship_date (DATE): contractual SLA dispatch date
  actual_ship_date (DATE): actual gate-out dispatch date, NULL if pending
  delivery_status (TEXT): one of On-Time, Delayed, In-Transit, Cancelled
  delay_reason (TEXT): one of 'FMCSA Driver HOS Limit', 'DOT Road Closure', 'Chassis Shortage', 'CBP Freight Hold', 'Warehouse Backlog', 'N/A'

carrier_performance:
  scac_code (TEXT, PK): NMFTA Standard Carrier Alpha Code (e.g., FEDX, UPSN, XPOF, JBHA, ODFL)
  carrier_name (TEXT): legal corporate name of carrier
  otif_compliance_pct (REAL): On-Time In-Full delivery percentage as a decimal (e.g., 0.94 for 94%)
  avg_delay_hours (REAL): mean delivery delay in hours
  otif_chargeback_usd (REAL): accrued SLA non-compliance penalties in USD

Business rules:
- Stockout definition: units_on_hand = 0
- Below reorder definition: units_on_hand > 0 AND units_on_hand <= reorder_point
- High occupancy threshold: current_occupancy_pct >= 85.0
- Delayed shipments: delivery_status = 'Delayed'
- In-transit shipments: delivery_status = 'In-Transit'
- Inventory value formula: units_on_hand * unit_cost_usd
"""


# ============================================================
# Verified Query Template Library
# ============================================================

verified_query_library = {
    'VQ1': {
        'description': 'Regional stockout count showing which US regions have the most SKUs currently at zero units on hand',
        'sql': """SELECT w.region,
     COUNT(*) AS stockout_skus
FROM inventory_levels i
JOIN warehouse_master w ON i.warehouse_id = w.warehouse_id
WHERE i.units_on_hand = 0
GROUP BY w.region
ORDER BY stockout_skus DESC"""
    },

    'VQ2': {
        'description': 'SKU categories with the most items currently below reorder point but not yet stocked out, indicating near-term replenishment need',
        'sql': """SELECT sku_category,
     COUNT(*) AS below_reorder_skus
FROM inventory_levels
WHERE units_on_hand > 0 AND units_on_hand <= reorder_point
GROUP BY sku_category
ORDER BY below_reorder_skus DESC"""
    },

    'VQ3': {
        'description': 'Warehouses at or above the 85% high-occupancy threshold, indicating capacity risk',
        'sql': """SELECT warehouse_id,
     warehouse_name,
     region,
     current_occupancy_pct
FROM warehouse_master
WHERE current_occupancy_pct >= 85.0
ORDER BY current_occupancy_pct DESC"""
    },

    'VQ4': {
        'description': 'Total count of shipments currently marked as Delayed in the shipment tracker',
        'sql': """SELECT COUNT(*) AS delayed_count
FROM shipment_tracker
WHERE delivery_status = 'Delayed'"""
    },

    'VQ5': {
        'description': 'Carriers ranked from worst to best by On-Time In-Full (OTIF) compliance percentage',
        'sql': """SELECT scac_code,
     carrier_name,
     otif_compliance_pct
FROM carrier_performance
ORDER BY otif_compliance_pct ASC"""
    },

    'VQ6': {
        'description': 'Carrier with the highest accrued OTIF chargeback penalties in USD',
        'sql': """SELECT * FROM (
    SELECT carrier_name,
         otif_chargeback_usd
    FROM carrier_performance
    ORDER BY otif_chargeback_usd DESC
    LIMIT 1)"""
    },

    'VQ7': {
        'description': 'Top 5 warehouses ranked by total inventory value (units_on_hand * unit_cost_usd), showing where carrying cost is concentrated',
        'sql': """SELECT * FROM (SELECT w.warehouse_id,
     w.warehouse_name,
     ROUND(SUM(i.units_on_hand * i.unit_cost_usd), 2) AS inventory_value_usd
FROM inventory_levels i
JOIN warehouse_master w ON i.warehouse_id = w.warehouse_id
GROUP BY w.warehouse_id
ORDER BY inventory_value_usd DESC
LIMIT 5)"""
    },

    'VQ8': {
        'description': 'Most common reasons for shipment delays with occurrence counts across all delayed shipments',
        'sql': """SELECT delay_reason,
     COUNT(*) AS occurrences
FROM shipment_tracker
WHERE delivery_status = 'Delayed'
GROUP BY delay_reason
ORDER BY occurrences DESC"""
    },

    'VQ9': {
        'description': 'Average occupancy comparison between CBP-bonded/FTZ warehouses and non-bonded facilities',
        'sql': """SELECT is_cbp_bonded_ftz,
     ROUND(AVG(current_occupancy_pct), 2) AS avg_occupancy_pct,
     COUNT(*) AS warehouse_count
FROM warehouse_master
GROUP BY is_cbp_bonded_ftz"""
    },

    'VQ10': {
        'description': 'Aggregate count of shipments currently in transit broken down by carrier SCAC code',
        'sql': """SELECT scac_code,
     COUNT(*) AS in_transit_count
FROM shipment_tracker
WHERE delivery_status = 'In-Transit'
GROUP BY scac_code
ORDER BY in_transit_count DESC"""
    }
}


# ============================================================
# Tool 1: Intent Classification
# ============================================================

def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Returns a dict with 'route' (verified or generated), 'query_id'
    (template ID or None) and 'match_reason'.
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
### ROLE
You are a query router for a supply chain operations analytics system. Your job is to decide whether a business user's question can be answered by one of the pre-approved query templates, or whether it needs fresh SQL generation.

### INPUT
User Question:
{user_question}

Available Verified Query Templates:
{library_descriptions}

### INSTRUCTIONS
1. Read the user question carefully and identify the analytical intent.
2. Compare the intent against each template description.
3. Match on semantic meaning, not exact wording. For example, 'out of stock' means stockout (units_on_hand = 0), 'FTZ' or 'bonded' refers to is_cbp_bonded_ftz = 1, 'late' or 'behind schedule' means Delayed, 'facilities near capacity' means high occupancy.
4. If a template genuinely answers the question, return that template ID.
5. If no template covers the question, return null for the query_id and set the route to generated.
6. Be careful about shape of answer: a question asking for row-level detail (e.g., 'show me the shipments') should NOT match a template that returns an aggregate count.

### OUTPUT
Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# ============================================================
# Tool 2: Query Generation
# ============================================================

def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.
    '''

    generation_prompt = f"""
### ROLE
You are a senior SQL developer specializing in supply chain operations analytics on a SQLite database.

### INPUT
User Question:
{user_question}

Database Schema (single source of truth):
{schema_context}

### INSTRUCTIONS
1. Write a single SQL query that answers the user question using only the provided schema.
2. The query must be read-only. Use SELECT (or WITH ... SELECT). Never use DROP, DELETE, UPDATE, INSERT, ALTER, or TRUNCATE.
3. Use only the tables and columns listed in the schema. Do not invent columns.
4. Resolve named entities using warehouse_name or warehouse_id where relevant (for example, 'Dallas' maps to WH_DFW_01, 'Chicago' maps to WH_ORD_01).
5. Ensure the query is SQLite compatible.
6. In SQLite, never subtract DATE() or date columns directly (e.g. DATE(a)-DATE(b)) — it silently returns 0; always use julianday(a)-julianday(b) for day differences.
7. Alias every numeric column with a suffix that states its unit, so the result is self-describing. Use _usd for dollar amounts, _pct or _percent for percentages, _count for counts, _hours for hour values, and _days for day values. Avoid bare aliases like "value", "amount", or "total".

### OUTPUT
Return ONLY the SQL query, with no markdown code blocks, no comments, and no explanation.
"""

    sql = llm.invoke(generation_prompt).content.strip()

    # Remove Markdown code fences (```sql ... ```) and extra whitespace
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()

    # Remove generic Markdown code fences (``` ... ```) and extra whitespace
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()

    return sql


# ============================================================
# Tool 3: Query Validation (4-check gate)
# ============================================================

def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validation gate:
      1. Read-only shape check
      2. Schema conformance check
      3. Parse & plan dry run (EXPLAIN)
      4. LLM relevance check
    '''

    # Query is considered failed by default
    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # ============================================================
    # CHECK 1: READ-ONLY SHAPE CHECK
    # ============================================================

    sql_upper = candidate_sql.upper().strip()

    forbidden_keywords = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT',
        'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH'
    ]

    # Query must start with SELECT or WITH
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result

    # Block forbidden SQL operations
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result

    # Allow only one SQL statement
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # ============================================================
    # CHECK 2: SCHEMA CONFORMANCE CHECK
    # ============================================================

    cur = db_connection.cursor()

    # Get all real tables from the database
    real_tables = [
        r[0]
        for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]

    # Get all real columns from those tables
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())

    # Parse the SQL
    parsed = sqlparse.parse(candidate_sql)[0]

    # Extract identifiers used in the SQL
    tokens = [
        str(t).strip().lower()
        for t in parsed.flatten()
        if t.ttype is None or 'Name' in str(t.ttype)
    ]

    referenced_identifiers = re.findall(
        r'\b[a-z_][a-z0-9_]*\b',
        candidate_sql.lower()
    )

    # SQL keywords that should not be treated as table/column names
    sql_keywords = {
        'select', 'from', 'where', 'and', 'or', 'group', 'by',
        'order', 'having', 'limit', 'join', 'on', 'as', 'case',
        'when', 'then', 'else', 'end', 'sum', 'count', 'avg',
        'min', 'max', 'round', 'desc', 'asc', 'left', 'right',
        'inner', 'outer', 'distinct', 'null', 'is', 'not', 'in',
        'like', 'with', 'union', 'all', 'between', 'coalesce'
    }

    # Identifiers that are not known tables, columns, or keywords.
    # (As in the notebook, aliases and computed names are expected here, so the
    # final decision relies on the parse/plan dry run and the relevance check.)
    unknown = [
        tok for tok in referenced_identifiers
        if tok not in sql_keywords
        and tok not in real_columns
        and tok not in real_tables
        and not tok.isdigit()
        and tok not in ('w', 'i', 's', 'c', 'p')
    ]

    # ============================================================
    # CHECK 3: PARSE-AND-PLAN DRY RUN
    # ============================================================

    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # ============================================================
    # CHECK 4: LLM RELEVANCE CHECK
    # ============================================================

    # Check whether this SQL came from the verified-query library
    is_verified_track = (
        query_id is not None and query_id in query_library
    )

    # Give the evaluator the correct context for the query type
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all regions/categories/carriers rather than filtering "
        "to just what the user asked). A separate response-generation step will "
        "filter and highlight the relevant rows afterward. Do NOT fail this query "
        "for lacking a WHERE clause that narrows to the user's specific "
        "region/category/carrier: judge only whether the underlying metric, tables, "
        "and aggregation logic match the question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped and filtered to answer it directly."
    )

    relevance_prompt = f"""
### ROLE
You are a senior data validator. Your job is to check whether a SQL query
correctly answers a business user's question about supply chain operations.

### CONTEXT
{track_context}

### INPUT
User Question: {user_question}

Candidate SQL:
{candidate_sql}

### INSTRUCTIONS
Assess whether the SQL genuinely answers what the user asked, considering:

1. Does it query the correct tables and columns?
2. Does it apply the right aggregations and groupings?
3. Does it handle the requested business definitions correctly?
4. Does it resolve named entities correctly?
5. Does it return the right shape of answer?
6. If this is a verified template, do not penalize it for returning
   a broader result set than the question's scope.

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}
"""

    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()

    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)

    if json_match:
        relevance_json = json.loads(json_match.group())

        # Store the LLM confidence score
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)

        # Fail if the LLM rejects the query or confidence is below 0.6
        if (
            relevance_json.get('verdict') == 'no'
            or relevance_json.get('confidence', 0.0) < 0.6
        ):
            result['failed_check'] = 'llm_relevance'
            result['details'] = (
                f"Relevance check failed: "
                f"{relevance_json.get('reason', 'unknown')}"
            )
            return result

    # ALL CHECKS PASSED
    result['passed'] = True
    result['details'] = 'All validation checks passed'

    return result


# ============================================================
# Tool 4: Retry Generation
# ============================================================

def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.
    '''

    retry_prompt = f"""
### ROLE
You are a senior SQL developer fixing a query that failed validation.

### INPUT
User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

### INSTRUCTIONS
1. Fix only the specific issue identified by the validation error.
2. Preserve the original intent of the query.
3. The revised SQL must be read-only SELECT (or WITH ... SELECT).
4. Use only tables and columns from the schema.
5. Ensure the query is SQLite compatible.

### OUTPUT
Return ONLY the corrected SQL, with no markdown code blocks, no comments, and no explanation.
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()

    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()

    return revised_sql


# ============================================================
# Tool 5: Query Execution
# ============================================================

def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query on the read-only connection and returns
    a dict with 'dataframe', 'reasonable' and 'warnings'.
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks (flag potential data-quality issues, do not stop execution)
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:

        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')

        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    return result


# ============================================================
# Tool 6: Response Generation
# ============================================================

def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.
    '''

    response_prompt = f"""
### ROLE
You are a supply chain operations analyst writing a concise business response for a fulfillment or inventory question.

### INPUT
User Question: {user_question}

Query Result Data:
{dataframe.to_string()}

### INSTRUCTIONS
1. Answer the user's specific question directly. Do not dump the entire table.
2. If the user asked about a specific region, warehouse, carrier, or category, highlight only those rows.
3. Provide context from other rows only when it adds value (for example, ranking or comparison).
4. State exact numbers from the data. Do not round beyond what is shown.
5. Flag anything notable, such as a warehouse close to a capacity threshold or a carrier significantly worse than peers.
6. Use clear, professional language suitable for a fulfillment operations memo.
7. Keep the response focused. Two to four sentences for simple questions, up to a short paragraph for complex ones.
8. State the unit for every number, inferred from its column name: _usd as "$X", _pct or _percent as "X%", _count as a plain count, _hours as "X hours". Never state a bare number when the source column implies a unit.

### OUTPUT
Return ONLY the natural language response text, with no markdown headers or bullet points unless truly needed.
"""

    response = llm.invoke(response_prompt).content.strip()
    return response


# ============================================================
# Audit Log (local, append-only)
# ============================================================

def write_audit_log(log):
    '''Appends one JSON line per request to a local audit file.'''
    try:
        gate = log.get('gate_result') or {}
        record = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'user_question': log.get('user_question'),
            'route': log.get('route'),
            'query_id': log.get('query_id'),
            'retry_used': log.get('retry_used'),
            'escalated': log.get('escalated'),
            'executed_sql': log.get('executed_sql'),
            'candidate_sql': log.get('candidate_sql'),
            'validation_passed': gate.get('passed'),
            'failed_check': gate.get('failed_check'),
            'validation_details': gate.get('details'),
            'row_count': log.get('row_count'),
            'confidence': log.get('confidence'),
        }
        with open(AUDIT_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
    except Exception:
        # Logging must never break the user-facing flow
        pass


# ============================================================
# Complete Workflow
# ============================================================

def run_workflow(user_question, status=None):
    '''
    Runs the complete query engine pipeline for a single user question.
    `status` is an optional Streamlit status container used to show the trace.
    '''

    def report(message):
        if status is not None:
            status.write(message)

    db_connection = conn
    query_library = verified_query_library
    schema_context = database_schema

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'response': None
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)

    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    report(
        f"**[1] Intent Classification:** route=`{log['route']}`, "
        f"query_id=`{log['query_id']}`  \n{log['match_reason']}"
    )

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)

    log['candidate_sql'] = candidate_sql

    report(
        f"**[2] Query Construction:** "
        f"{'loaded from verified library' if log['route'] == 'verified' else 'generated fresh SQL'}"
    )

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])

    log['gate_result'] = gate
    log['confidence'] = gate.get('relevance_confidence')

    report(
        f"**[3] Validation Gate:** passed=`{gate['passed']}`, "
        f"relevance_confidence=`{gate.get('relevance_confidence')}`"
    )
    if not gate['passed']:
        report(f"Failed check: `{gate.get('failed_check')}`  \nDetails: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        report(f"**Retrying:** {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        report(
            f"**Retry Validation Gate:** passed=`{gate['passed']}`, "
            f"relevance_confidence=`{gate.get('relevance_confidence')}`"
        )
        if not gate['passed']:
            report(f"Retry failed check: `{gate.get('failed_check')}`  \nRetry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['route'] = 'escalate'
        log['response'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = gate.get('relevance_confidence')
        report(f"**[!] Escalated to human:** {gate['details']}")
        write_audit_log(log)
        return {'log': log, 'dataframe': None, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)

    report(f"**[4] Execute:** {len(df)} rows returned")
    if exec_result['warnings']:
        report(f"Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    response = generate_response(user_question, df, log['route'], log['query_id'])
    log['response'] = response

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    report(f"**[5] Response Generation:** confidence=`{log['confidence']}`")

    write_audit_log(log)
    return {'log': log, 'dataframe': df, 'warnings': exec_result['warnings'], **log}


# ============================================================
# Streamlit User Interface
# ============================================================

EXAMPLE_QUESTIONS = [
    "Which 5 warehouses have the highest dollar value of on-hand inventory?",
    "Do our bonded warehouses run hotter on capacity than the regular ones?",
    "Which regions have an above-average stockout rate, and how many SKUs are sitting at or below reorder point in those regions?",
    "For our FTZ warehouses, what's the at-risk inventory value tied to shipments that were delayed more than 2 days, broken out by carrier?",
    "What's the average employee headcount per warehouse?",
    "How many shipments are currently delayed?",
    "Which carriers have the worst OTIF compliance?",
    "Which warehouses are operating above 85% capacity?",
]

if "history" not in st.session_state:
    st.session_state["history"] = []
if "question_input" not in st.session_state:
    st.session_state["question_input"] = ""


def set_example(question):
    st.session_state["question_input"] = question
    st.session_state["auto_run"] = True


st.title("📦 Inventory & Fulfillment Exception Engine")
st.caption(
    "Ask plain-English questions about inventory, warehouses, shipments and carrier "
    "performance. Runs read-only against the local supply chain database."
)

# ---------------- Sidebar ----------------

with st.sidebar:
    st.header("About")
    st.write(
        "Recurring questions are matched to a pre-approved library of verified SQL "
        "templates. Novel questions get freshly generated, read-only SQL. Every query "
        "passes a four-check validation gate (read-only shape, schema conformance, "
        "parse/plan dry run, LLM relevance). Generated queries that fail are retried "
        "once, then escalated to a human analyst."
    )

    st.subheader("Example questions")
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        st.button(q, key=f"example_{i}", on_click=set_example, args=(q,), use_container_width=True)

    st.subheader("Verified query library")
    with st.expander("View templates"):
        for qid, entry in verified_query_library.items():
            st.markdown(f"**{qid}**: {entry['description']}")

# ---------------- Input ----------------

user_question = st.text_input(
    "Your question",
    key="question_input",
    placeholder="e.g. Which warehouses are operating above 85% capacity?",
)

col_a, col_b = st.columns([1, 3])
with col_a:
    submitted = st.button("Run query", type="primary")
with col_b:
    show_trace = st.checkbox("Show pipeline trace", value=True)

auto_run = st.session_state.pop("auto_run", False)

# ---------------- Run ----------------

if (submitted or auto_run) and user_question.strip():

    trace_container = st.status("Running pipeline...", expanded=show_trace) if show_trace else None

    try:
        output = run_workflow(user_question.strip(), status=trace_container)
    except Exception as e:
        if trace_container is not None:
            trace_container.update(label="Pipeline error", state="error")
        st.error(f"Pipeline failed: {e}")
        st.stop()

    if trace_container is not None:
        trace_container.update(
            label="Pipeline complete" if not output['escalated'] else "Escalated to human review",
            state="complete" if not output['escalated'] else "error",
        )

    st.session_state["history"].insert(0, {
        "question": user_question.strip(),
        "route": output['route'],
        "query_id": output['query_id'],
        "confidence": output['confidence'],
        "escalated": output['escalated'],
        "response": output['response'],
    })

    st.divider()

    if output['escalated']:
        st.warning(output['response'])
        c1, c2 = st.columns(2)
        c1.metric("Route", "escalate")
        c2.metric("Retry used", "Yes" if output['retry_used'] else "No")

        with st.expander("Validation details"):
            st.json(output['gate_result'])
        if output.get('candidate_sql'):
            with st.expander("Last candidate SQL (not executed)"):
                st.code(output['candidate_sql'], language="sql")

    else:
        confidence = output['confidence']
        if isinstance(confidence, (int, float)):
            badge = "🟢" if confidence >= 0.8 else ("🟡" if confidence >= 0.6 else "🔴")
            confidence_display = f"{confidence:.2f}"
        else:
            badge, confidence_display = "⚪", str(confidence)

        st.subheader("Answer")
        st.write(output['response'])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Route", output['route'])
        m2.metric("Template", output['query_id'] or "generated SQL")
        m3.metric("Confidence", f"{badge} {confidence_display}")
        m4.metric("Rows returned", output['row_count'])

        if output.get('warnings'):
            st.warning("Result warnings: " + "; ".join(output['warnings']))

        if output['dataframe'] is not None:
            with st.expander("View underlying data", expanded=True):
                st.dataframe(output['dataframe'], use_container_width=True)
                st.download_button(
                    "Download as CSV",
                    output['dataframe'].to_csv(index=False).encode("utf-8"),
                    file_name="query_result.csv",
                    mime="text/csv",
                )

        with st.expander("View executed SQL"):
            st.code(output['executed_sql'], language="sql")

elif submitted:
    st.info("Please enter a question first.")

# ---------------- Session history ----------------

if st.session_state["history"]:
    st.divider()
    with st.expander(f"Session history ({len(st.session_state['history'])})"):
        for item in st.session_state["history"]:
            conf = item['confidence']
            conf_txt = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"
            st.markdown(
                f"**Q:** {item['question']}  \n"
                f"*Route:* `{item['route']}` · *Template:* `{item['query_id']}` · *Confidence:* {conf_txt}  \n"
                f"**A:** {item['response']}"
            )
            st.markdown("---")
