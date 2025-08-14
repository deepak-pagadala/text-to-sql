import os
import re
import json
import sqlite3
import faiss
import torch                    
import numpy as np
import pandas as pd
from openai import OpenAI
from functools import lru_cache
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# LangChain memory imports
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.llms.base import LLM
from typing import Optional, List, Dict, Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"  

# -------------------------------------------------------------------------- #
# 1.  ENVIRONMENT VARIABLES & PATHS
# -------------------------------------------------------------------------- #
load_dotenv()                     

HF_TOKEN        = os.getenv("HF_TOKEN")       
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") 

if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY not set in .env")

DB_PATH          = "chatbot22.db"
NTEE_PATH        = "NTEE_descriptions.csv"
SAMPLEQUERY_PATH = "sample_query.jsonl"
ROW_THRESHOLD = 500          
TOP_K_DEFAULT = 10

# -------------------------------------------------------------------------- #
# 2.  CONNECT TO MISTRAL-LARGE + LANGCHAIN WRAPPER                                    
# -------------------------------------------------------------------------- #
client = OpenAI(
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1",   
)

MISTRAL_MODEL_ID = "mistral-large-latest"

class MistralLLM(LLM):
    """LangChain-compatible wrapper for Mistral API"""
    
    @property
    def _llm_type(self) -> str:
        return "mistral"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = client.chat.completions.create(
            model=MISTRAL_MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

# Initialize LangChain components
mistral_llm = MistralLLM()
memory = ConversationSummaryBufferMemory(
    llm=mistral_llm,
    max_token_limit=1000,
    return_messages=True
)

def mistral_generate(prompt: str, max_new: int = 256) -> str:
    """
    Call Mistral Large and return the assistant's reply as plain text.
    """
    response = client.chat.completions.create(
        model=MISTRAL_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# -------------------------------------------------------------------------- #
# MEMORY-AWARE QUERY EXPANSION
# -------------------------------------------------------------------------- #
def expand_query_with_context(user_query: str) -> str:
    """
    Use conversation memory to expand incomplete/contextual queries.
    Returns the original query if no context is needed, or an expanded version.
    """
    # Get conversation history
    chat_history = memory.chat_memory.messages
    
    if not chat_history:
        return user_query
    
    # Check if query seems incomplete/contextual
    contextual_indicators = [
        r'\band\s+(greater|less|more|fewer)',
        r'\bwhat\s+about\b',
        r'\bhow\s+about\b', 
        r'\band\s+(those|these|them)',
        r'^\s*(and|or|but)\s+',
        r'\bthe\s+opposite\b',
        r'\bconversely\b',
        r'\bin\s+contrast\b'
    ]
    
    is_contextual = any(re.search(pattern, user_query, re.I) for pattern in contextual_indicators)
    
    if is_contextual:
        # Get recent context (last 2-3 exchanges)
        recent_messages = chat_history[-6:] if len(chat_history) > 6 else chat_history
        context_text = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in recent_messages
        ])
        
        expansion_prompt = f"""
Given this conversation context:
{context_text}

The user just asked: "{user_query}"

This seems to be a follow-up question that needs context. Rewrite it as a complete, standalone question that incorporates the relevant context from the conversation.

Examples:
- "and greater than?" → "how many organizations gave grants greater than $50000 in the year 2019?"
- "what about healthcare?" → "which foundations support healthcare projects?"

Return only the expanded question, nothing else:
"""
        
        try:
            expanded = mistral_generate(expansion_prompt, max_new=128)
            print(f"🧠 Expanded with context: '{expanded}'")
            return expanded.strip().strip('"').strip("'")
        except Exception as e:
            print(f"⚠️ Context expansion failed: {e}")
    
    return user_query

# -------------------------------------------------------------------------- #
# 2b.  Back-compat helper 
# -------------------------------------------------------------------------- #
llama_generate = mistral_generate     

# -------------------------------------------------------------------------- #
# 3.  OPEN SQLITE DB & LIST TABLES  
# -------------------------------------------------------------------------- #
with sqlite3.connect(DB_PATH) as conn:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    ALL_TABLES = [row[0] for row in cur.fetchall()]

print("Core initialisation complete.  Tables detected:", ", ".join(ALL_TABLES))

# [Keep all the existing code sections: COLUMN_MAPPING, RESTRICTED_COLUMNS, etc.]
COLUMN_MAPPING = {
    "research_org_pseudonym": "foundation_giver_id",
    "org_name": "organization_name",
    "org_city": "organization_city",
    "org_state": "organization_state",
    "org_zip": "organization_zipcode",
    "org_county_code": "organization_county_code",
    "org_ntee_code": "organization_ntee_code",
    "org_classification_desc_short": "org_classification_desc_short",
    "org_classification_desc_medium": "org_classification_desc_medium",
    "org_num_employees": "number_of_employees_in_organization",
    "org_num_volunteers": "number_of_volunteers_in_organization",
    "org_total_expenses": "total_expenses_of_organization",
    "org_assets": "organization_assets",
    "org_liabilities": "organization_liabilities",
    "org_total_revenue": "total_revenue_of_organization",
    "org_type": "organization_type",
    "org_subtype": "organization_subtype",
    "org_date_established": "date_when_organization_was_established",
    "org_naics_code": "organization_naics_code",
    "org_address": "organization_address",
    "org_latitude": "organization_latitude",
    "org_longitude": "organization_longitude",
    "org_financial_year_start": "financial_start_year_of_organization",
    "org_financial_year_end": "financial_end_year_of_organization",
    "org_revenue_contributions": "organization_revenue_contributions",
    "org_mission_statement_summary": "organization_mission_statement_summary",
    "org_source": "organization_source",
    "org_government_revenue": "organization_government_revenue",
    "org_ntee_major10": "organization_nteecode_first_letter",
    "org_county_name": "organization_county_name",
    "org_rurality_percentage": "organization_rurality_percentage",
    "org_county_class": "organization_county_class",
    "giver_id": "foundation_giver_id",
    "grant_date": "grant_date",
    "grant_amount": "grant_amount",
    "g_city": "grantee_city",
    "g_state": "grantee_state",
    "g_zip": "grantee_zipcode",
    "g_ein": "grantee_ein",
    "g_name": "grantee_name",
    "f_city": "foundation_city",
    "f_state": "foundation_state",
    "f_zip": "foundation_zipcode",
    "f_state_name": "foundation_state_name",
    "f_region_4": "foundation_region_4",
    "f_region_9": "foundation_region_9",
    "g_state_name": "grantee_state_name",
    "g_region_4": "grantee_region_4",
    "g_region_9": "grantee_region_9",
    "f_ein": "foundation_ein",
    "f_name": "foundation_name",
    "grant_source": "grant_source",
    "formation_yr": "formation_year",
    "ruling_yr": "ruling_year",
    "f_mission_partI": "foundation_mission_partI",
    "f_mission_partIII": "foundation_mission_partIII",
    "g_mission_partI": "grantee_mission_partI",
    "g_mission_partIII": "grantee_mission_partIII",
    "f_ntee_letter": "foundation_ntee_code",
    "g_ntee_letter": "grantee_ntee_code",
    "f_ntee_description": "foundation_ntee_description",
    "g_ntee_description": "grantee_ntee_description",
    "f_ntee_major8": "foundation_ntee_short_description",
    "f_ntee_major10": "foundation_ntee_description_abbreviation",
    "f_ntee_major12": "foundation_ntee_major12",
    "g_ntee_major8": "grantee_ntee_short_description",
    "g_ntee_major10": "grantee_ntee_description_abbreviation",
    "g_ntee_major12": "grantee_ntee_major12",
    "f_amt_assets_total": "foundation_assets_total",
    "f_amt_exp_grants": "foundation_amount_exp_grants",
    "f_num_employees": "foundation_number_of_employees",
    "f_num_volunteers": "foundation_number_of_volunteers",
    "g_amt_assets_total": "grantee_assets_total",
    "g_amt_exp_total": "grantee_amount_exp_total",
    "g_num_employees": "grantee_number_of_employees",
    "g_num_volunteers": "grantee_number_of_volunteers",
    "g_amt_rev_contrib_total": "grantee_revenue_contribution_total",
    "g_amt_rev_total": "grantee_revenue_total",
}

RESTRICTED_COLUMNS = [
    "grantee_ntee_description",
    "foundation_ntee_description",
    "grantee_ntee_major12",
    "grantee_ntee_description_abbreviation",
    "foundation_ntee_major12",
    "foundation_ntee_description_abbreviation",
    "foundation_ntee_short_description",
    "grantee_ntee_short_description",
    "organization_nteecode_first_letter",
]

TABLE_MAPPING = {
    # legacy / generic names the LLM might output → actual SQLite tables
    "user_data":          "RRNA_Org_Data",
    "organization_data":  "RRNA_Org_Data",
    "organizations":      "RRNA_Org_Data",
    "orgs":               "RRNA_Org_Data",
    "grants":             "RRNAResearchGrants",
    "research_grants":    "RRNAResearchGrants",
}

def replace_table_aliases(sql: str) -> str:
    """
    Swap out any alias in TABLE_MAPPING with the real table
    *before* we send the query to SQLite.
    """
    for alias, real in TABLE_MAPPING.items():
        sql = re.sub(rf"\b{alias}\b", real, sql, flags=re.I)
    return sql

@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load 'all-MiniLM-L6-v2' once and cache it."""
    print("Loading MiniLM embedder")
    return SentenceTransformer("all-MiniLM-L6-v2")

EMBEDDER = get_embedder()

# ------------------------------------------------------------------- #
#  DEBUG SWITCH 
# ------------------------------------------------------------------- #
DEBUG_SQL = False       

def _dbg(label: str, msg: str):
    """Lightweight debug printer controlled by DEBUG_SQL flag."""
    if DEBUG_SQL:
        print(f"\n🔍  {label}: {msg}\n")

# [Keep all existing helper functions unchanged]
def get_ntee_code_from_csv(user_query: str, csv_path: str = NTEE_PATH):
    """
    Semantic search over NTEE descriptions.
    Returns (ntee_code, short_description).
    """
    df = pd.read_csv(csv_path, encoding="latin1")

    required = {"org_class_code", "class_short_description"}
    if not required.issubset(df.columns):
        raise ValueError(
            "CSV must contain 'org_class_code' and 'class_short_description'."
        )

    desc_embeddings = EMBEDDER.encode(df["class_short_description"].tolist())
    user_embedding  = EMBEDDER.encode([user_query])

    # cosine similarity
    desc_norm  = desc_embeddings / np.linalg.norm(desc_embeddings, axis=1, keepdims=True)
    query_norm = user_embedding / np.linalg.norm(user_embedding)
    sims       = np.dot(desc_norm, query_norm.T).squeeze()

    best_idx   = int(np.argmax(sims))
    return (
        df.iloc[best_idx]["org_class_code"],
        df.iloc[best_idx]["class_short_description"],
    )

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee legacy column names so the rest of the codebase (which
    expects 'Question' and 'SQL Query') doesn't have to change.
    """
    rename_map = {
        "question":   "Question",
        "Question":   "Question",
        "sql_query":  "SQL Query",
        "SQL Query":  "SQL Query",
        "sql":        "SQL Query",
    }
    return df.rename(columns=rename_map, errors="ignore")

# [Keep all classification functions unchanged]
def classify_query(
    query: str, positive_examples: list[str], negative_examples: list[str]
) -> bool:
    """
    Returns True if the query is closer to the positive centroid than the negative.
    """
    pos_emb = EMBEDDER.encode(positive_examples)
    neg_emb = EMBEDDER.encode(negative_examples)

    pos_cent = pos_emb.mean(axis=0)
    neg_cent = neg_emb.mean(axis=0)

    q_vec = EMBEDDER.encode([query])[0]
    q_vec /= np.linalg.norm(q_vec)

    sim_pos = np.dot(q_vec, pos_cent / np.linalg.norm(pos_cent))
    sim_neg = np.dot(q_vec, neg_cent / np.linalg.norm(neg_cent))

    return sim_pos > sim_neg

def is_organization_query(query: str) -> bool:
    org_examples = [
        "Which foundations have provided the highest grants?",
        "Tell me about the foundations giving grants.",
        "List organizations that offer grants.",
        "What organizations support environmental projects?",
    ]
    non_org_examples = [
        "What is the weather in New York?",
        "Show me the sales numbers for Q3.",
        "How many users signed up today?",
    ]
    return classify_query(query, org_examples, non_org_examples)

def is_grantee_query(query: str) -> bool:
    grantee_examples = [
        "Which grantees received the highest funding?",
        "Tell me about the grantees that got the most grants.",
        "List grantees who work in education.",
        "Which organizations have received the most grants?",
    ]
    non_grantee_examples = [
        "Which organizations gave the highest grants?",
        "Show foundations supporting healthcare.",
        "How many users signed up today?",
    ]
    return classify_query(query, grantee_examples, non_grantee_examples)

def is_sector_related_query(query: str) -> bool:
    sector_examples = [
        "Which foundations support education?",
        "Give me organizations in the healthcare sector.",
        "List tech-focused grant providers.",
    ]
    non_sector_examples = [
        "Which organizations gave the highest grants?",
        "How many companies donated over $10 million?",
        "List organizations by their total grants.",
    ]
    return classify_query(query, sector_examples, non_sector_examples)

# [Keep all existing functions: fetch_schema_from_db, create_faiss_index, etc.]
def fetch_schema_from_db(db_path: str = DB_PATH) -> dict[str, list[str]]:
    """
    Return {"table_name": [mapped_column1, mapped_column2, ...], …}
    using COLUMN_MAPPING to make names human-readable.
    """
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cur.fetchall()]

    schemas: dict[str, list[str]] = {}
    for table_name in tables:
        cur.execute(f'PRAGMA table_info("{table_name}");')
        raw_cols = [row[1] for row in cur.fetchall()]
        mapped   = [COLUMN_MAPPING.get(col, col) for col in raw_cols]
        schemas[table_name] = mapped

    conn.close()
    return schemas

def create_faiss_index(schema_texts: list[str], index_path: str = "index_file.index"):
    """
    Build a simple L2 FAISS index of MiniLM embeddings for schema strings
    (column names, table descriptions, etc.).  Saves it to disk.
    """
    embeddings = EMBEDDER.encode(schema_texts)
    embeddings = embeddings.astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"✅  FAISS index saved to {index_path}  (vectors: {len(schema_texts)})")
    return index

def generate_multiple_sql_queries_with_schema(user_query: str, schemas: dict[str, list[str]]):
    """
    Use the DB schema + few-shot examples + optional NTEE enrichment.  Returns a list of ≤ 3 SQL strings.
    """
   
    schema_block = "\n".join(
        f"Table: {tbl} - Columns: {', '.join(cols)}" for tbl, cols in schemas.items()
    )

    if (is_organization_query(user_query) or is_grantee_query(user_query)) \
       and is_sector_related_query(user_query):
        code, desc = get_ntee_code_from_csv(user_query, NTEE_PATH)
        user_query = f"{user_query} (NTEE Code: {code} - {desc})"

    few_shot_examples = """
Example 1:
Question: Which foundations have given the highest amount of grants?
SQL Query: SELECT foundation_name, MAX(grant_amount) AS max_grant
           FROM your_table
           GROUP BY foundation_name
           ORDER BY max_grant DESC;

Example 2:
Question: Show the top 5 grantees in the healthcare sector.
SQL Query: SELECT grantee_name, SUM(grant_amount) AS total_grants
           FROM your_table
           WHERE sector LIKE '%health%'
           GROUP BY grantee_name
           ORDER BY total_grants DESC
           LIMIT 5;
""".strip()

    prompt = f"""
User Query:
{user_query}

Few-shot Examples:
{few_shot_examples}

Schema Details:
{schema_block}

Generate the most relevant SQL query for the given prompt. Ensure that the
query is syntactically correct and works with the provided schema.
If you have to compare something, use the LIKE operator. For state codes,
use abbreviations.  Do NOT use the following columns in any query:
{RESTRICTED_COLUMNS}.
Return ONLY the SQL query in a single line. Do not include any explanations.
""".strip()

    llm_output = llama_generate(prompt)

    sql_lines = [ln.strip() for ln in llm_output.splitlines() if ln.strip()]
    return sql_lines[:3]

# [Keep all existing functions: is_broad_query, quick_reframe, etc.]
TOP_K_DEFAULT  = 10                      
BROAD_KEYWORDS = (
    " all ", " every ", " list of all ", " show all ", " give me all "
)

def is_broad_query(q: str) -> bool:
    """
    Heuristic:
      • must contain an 'all/every/…' keyword   AND
      • must NOT contain an explicit row-limit hint
        — keywords  (top, first, limit …) OR
        — a small integer (≤ 50)
    Years like 2024 are *ignored*; they don't count as limits.
    """
    ql = " " + q.lower() + " "

   
    if not any(kw in ql for kw in BROAD_KEYWORDS):
        return False

    if re.search(r"\b(top|limit|first|last|largest|smallest|highest|lowest)\b", ql):
        return False

    small_nums = [int(n) for n in re.findall(r"\b\d+\b", ql) if int(n) <= 50]
    if small_nums:
        return False

    return True

def quick_reframe(q: str, k: int = TOP_K_DEFAULT) -> str:
    """
    Local (no-LLM) rewrite: 'all organisations …' → 'top-K organisations …'.
    """
    out = re.sub(r"\blist of all\b", f"top {k}", q, flags=re.I)
    out = re.sub(r"\ball\b",        f"top {k}", out, flags=re.I)
    if out == q:                                    # nothing matched
        out = f"Show the top {k} results for: {q}"
    return out

def _count_rows(conn: sqlite3.Connection, sql: str) -> int:
    """
    Estimate result size safely, even when the query has aliases, ORDER BY,
    GROUP BY, or LIMIT.  Strategy:

      SELECT COUNT(*) FROM ( <original_query_without_semicolon> ) AS sub;
    """
    # strip trailing semicolons / whitespace
    inner = sql.strip().rstrip(";").strip()

    wrapper = f"SELECT COUNT(*) FROM ({inner}) AS sub__cnt__;"
    try:
        cur = conn.execute(wrapper)
        return cur.fetchone()[0]
    except Exception:
        
        try:
            cur = conn.execute(inner + " LIMIT 1000000")
            return sum(1 for _ in cur)
        except Exception:
            return 10**9

def _reframe_with_llm(prompt: str,
                      row_cnt: int,
                      top_k: int = TOP_K_DEFAULT) -> str:
    """
    Ask Mistral to rewrite the user's request into a bounded version.
    """
    sys_inst = (
        "You are a helpful data assistant. Rewrite overly broad questions "
        "into concise alternatives that return at most {k} rows. "
        "Keep the new question in the same semantic domain, mention the limit "
        "explicitly ('top {k}', 'first {k}', etc.)."
    ).format(k=top_k)

    llm_prompt = (
        f"User asked: \"{prompt}\".\n\n"
        f"The original SQL would return about {row_cnt} rows "
        f"which is too many to display.\n"
        f"Rewrite the question so it returns at most {top_k} records."
    )

    response = client.chat.completions.create(
        model=MISTRAL_MODEL_ID,
        messages=[
            {"role": "system", "content": sys_inst},
            {"role": "user",   "content": llm_prompt},
        ],
        temperature=0.3,
        max_tokens=128,
    )
    return response.choices[0].message.content.strip()

def execute_sql_query(sql_query: str, db_path: str = DB_PATH):
    """
    Run `sql_query` on SQLite and return rows (or an error string).
    Ensures only **one** statement is sent to the engine.
    """
   
    if ";" in sql_query:
        sql_query = sql_query.split(";", 1)[0]

    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        return cursor.fetchall()
    except sqlite3.Error as exc:
        return f"An error occurred: {exc}"
    finally:
        conn.close()

def revert_aliases_in_query(sql_query: str) -> str:
    """Replace human-friendly column names with their DB originals."""
    reverse = {v: k for k, v in COLUMN_MAPPING.items()}
    for alias, original in reverse.items():
        sql_query = re.sub(rf"\b{alias}\b", original, sql_query)
    return sql_query

def clean_sql_query(query: str) -> str:
    """
    • Remove ``` fences / stray back-ticks
    • Keep **only the first SQL statement** (SQLite can run one at a time)
    • Trim trailing semicolons / whitespace
    """
    q = re.sub(r"`{3}sql|`", "", query).strip()
    if ";" in q:
        q = q.split(";", 1)[0]

    return q.strip()

def summarize_findings_with_llm(
    user_query: str,
    schema: dict[str, list[str]],
    results,
) -> str:
    """
    Produce a ~50-word explanation of `results`, grounded in `schema`.
    """
  
    schema_txt  = "\n".join(
        f"Table: {tbl} - Columns: {', '.join(cols)}" for tbl, cols in schema.items()
    )
    result_txt  = "\n".join(map(str, results))

   
    sector_hint = ""
    if results and isinstance(results[0], (list, tuple)) and len(results[0]) >= 3:
        maybe_sector = results[0][-1]
        if isinstance(maybe_sector, str) and len(maybe_sector) > 3:
            sector_hint = f"This query is related to the sector: '{maybe_sector}'.\n\n"

   
    prompt = f"""
User Query: {user_query}

Schema Details:
{schema_txt}

SQL Results:
{result_txt}

{sector_hint}Generate a concise, human-friendly summary of the results in about 50 words.
Be specific about the sector or topic.
""".strip()

    return llama_generate(prompt)

def save_query_sql_pair(question: str, sql: str,
                        jsonl_path: str = SAMPLEQUERY_PATH) -> None:
    """
    Append a (Q, SQL) pair to sample_query.jsonl iff it's not already in there,
    then clear the FAISS cache so the new example is live.
    """
    # read existing
    if os.path.exists(jsonl_path):
        df = pd.read_json(jsonl_path, lines=True)
        df = _ensure_cols(df)
    else:
        df = pd.DataFrame(columns=["Question", "SQL Query"])

    # duplicate?
    duplicate = ((df["Question"] == question) &
                 (df["SQL Query"] == sql)).any()
    if duplicate:
        return  # nothing to do

    # append to disk
    with open(jsonl_path, "a", encoding="utf-8") as f:
        json.dump({"question": question, "sql_query": sql}, f, ensure_ascii=False)
        f.write("\n")

    # bust the cache so next call sees the new row
    _load_sample_queries.cache_clear()

# [Keep all existing functions: run_pipeline, _load_sample_queries, etc.]
def run_pipeline(user_query: str, db_path: str = DB_PATH):
    """
    End-to-end: schema → prompt → (≤3) SQL → first query that returns rows →
                OPTIONAL row-count check & LLM reframing →
                enrichment → 50-word Mistral summary
    """
    schemas       = fetch_schema_from_db(db_path)
    candidate_sql = generate_multiple_sql_queries_with_schema(user_query, schemas)

    conn = sqlite3.connect(db_path)      
    try:
        for sql in candidate_sql:
            cleaned = clean_sql_query(sql)
            raw_sql = revert_aliases_in_query(cleaned)
            raw_sql = replace_table_aliases(raw_sql)

           
            row_cnt = _count_rows(conn, raw_sql)
            if row_cnt > ROW_THRESHOLD:
                print(
                    f"\nQuery would return ≈{row_cnt:,} rows "
                    f"(threshold = {ROW_THRESHOLD})."
                )

                alt_prompt = _reframe_with_llm(user_query, row_cnt, TOP_K_DEFAULT)
                print(f"🤖 Suggestion → {alt_prompt}")

                ans = input("   ▶ Use this reframed question? [y/N]: ").strip().lower()
                if ans == "y":
                   
                    alt_sqls = generate_multiple_sql_queries_with_schema(
                        alt_prompt, schemas
                    )
                    if alt_sqls:
                        cleaned   = clean_sql_query(alt_sqls[0])
                        raw_sql   = revert_aliases_in_query(cleaned)
                        user_query = alt_prompt        # for the summariser
                        print("… proceeding with reframed query.\n")
                else:
                    print("… keeping the original request.\n")
           
            _dbg("CLEANED SQL", raw_sql)
            results = execute_sql_query(raw_sql, db_path)

            if results and not isinstance(results, str):
               
                try:
                    df_ntee   = pd.read_csv(NTEE_PATH, encoding="latin1")
                    code2desc = dict(
                        zip(
                            df_ntee["org_class_code"],
                            df_ntee["class_short_description"]
                        )
                    )

                    code_idx = None
                    for i, val in enumerate(results[0]):
                        if isinstance(val, str) and re.match(r"^[A-Z]\d{2}", val):
                            if val in code2desc:
                                code_idx = i
                                break

                    if code_idx is not None:
                        results = [
                            tuple(list(row) + [code2desc.get(row[code_idx], "Unknown Sector")])
                            for row in results
                        ]
                except Exception:
                    pass  

                summary = summarize_findings_with_llm(user_query, schemas, results)

                try:
                    save_query_sql_pair(user_query, cleaned)
                except Exception as exc:
                    _dbg("SAVE-PAIR WARN", f"Could not save pair: {exc}")

                return cleaned, results, summary

    finally:
        conn.close()

    return "No valid results found for any query.", None, None

@lru_cache(maxsize=None)
def _load_sample_queries():
    """
    Load sample_query.jsonl → DataFrame → FAISS index.
    Called once per run (cached); cache is cleared whenever we append
    a new example so the fresh row is searchable immediately.
    """
    if not os.path.exists(SAMPLEQUERY_PATH):
        # start from an empty frame with canonical col-names
        df = pd.DataFrame(columns=["Question", "SQL Query"])
        return df, None, None          # (DF, faiss_index, embeddings)

    # 1) read the newline-delimited JSON
    df = pd.read_json(SAMPLEQUERY_PATH, lines=True)
    df = _ensure_cols(df)

    # 2) embed the questions
    questions = df["Question"].tolist()
    if not questions:
        return df, questions, None

    embeds = EMBEDDER.encode(questions).astype("float32")

    # 3) build / rebuild FAISS index (cosine ~ inner product on L2-normed vecs)
    index = faiss.IndexFlatIP(embeds.shape[1])
    faiss.normalize_L2(embeds)
    index.add(embeds)

    return df, questions, index

def get_top_similar_queries(user_query: str, k: int = 3):
    df, questions, index = _load_sample_queries()
    if index is None:
        return [], df
    
    q_embed = EMBEDDER.encode([user_query]).astype("float32")
    faiss.normalize_L2(q_embed)
    dist, idx = index.search(q_embed, k)
    tops = [(questions[i], float(d), i) for d, i in zip(dist[0], idx[0]) if i < len(questions)]
    return tops, df

# -------------------------------------------------------------------------- #
#  MEMORY-ENHANCED INTERACTIVE CLI                   
# -------------------------------------------------------------------------- #
def interactive_cli():
    """
    Enhanced CLI with conversation memory and context-aware query expansion.
    """
    print("────────────────────────────────────────────────────────────")
    print("  CharityBot – ask about grants, foundations or sectors")
    print("  🧠 Now with conversation memory!")
    print("  (type 'exit' to quit, 'memory' to see conversation history)")
    print("────────────────────────────────────────────────────────────")

    while True:
        try:
            user_q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  Bye!")
            break

        if user_q.lower() in {"exit", "quit"}:
            break
        
        if user_q.lower() == "memory":
            print("\n💭 Conversation History:")
            for i, msg in enumerate(memory.chat_memory.messages):
                speaker = "You" if isinstance(msg, HumanMessage) else "Bot"
                print(f"  {i+1}. {speaker}: {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
            continue

        # Add user message to memory
        memory.chat_memory.add_user_message(user_q)
        
        # Expand query with context if needed
        expanded_query = expand_query_with_context(user_q)
        
        # Use expanded query for similarity search
        tops, df = get_top_similar_queries(expanded_query)
        
        if tops:  # Only show suggestions if we have similar queries
            print("\nDid you mean:")
            for i, (q, score, _) in enumerate(tops, 1):
                print(f"  {i}. {q}    [similarity {score:.2f}]")
            print("  0. None of the above")

            choice = input("Select 0-3 and press Enter: ").strip()

            if choice.isdigit() and 1 <= int(choice) <= len(tops):
                # User selected a cached query
                _, _, idx   = tops[int(choice) - 1]
                canned_sql  = df.iloc[idx]["SQL Query"]
                schemas     = fetch_schema_from_db(DB_PATH)
                cleaned     = clean_sql_query(canned_sql)
                _dbg("CANNED SQL", cleaned)

                results     = execute_sql_query(cleaned, DB_PATH)
                summary     = summarize_findings_with_llm(expanded_query, schemas, results)
                
                print("\nBot:", summary)
                
                # Add bot response to memory
                memory.chat_memory.add_ai_message(summary)
                continue

        # Handle broad queries or proceed with pipeline
        if is_broad_query(expanded_query):
            alt_q = quick_reframe(expanded_query, TOP_K_DEFAULT)
            print(
                "\nThat request is likely to return too many rows."
                f"\nHow about:\n   ▶ {alt_q}"
                "\n1 = Yes, proceed  \n0 = No, I'll rephrase"
            )
            ans = input("Select 1 or 0 and press Enter: ").strip()
            if ans == "1":
                expanded_query = alt_q        
            else:
                print("Okay, please try a different question.")
                continue                 
        
        try:
            _, _, summary = run_pipeline(expanded_query, DB_PATH)
            if summary:
                print("\nBot:", summary)
                # Add bot response to memory
                memory.chat_memory.add_ai_message(summary)
            else:
                error_msg = "I couldn't find relevant results for that query. Please try rephrasing."
                print(f"\nBot: {error_msg}")
                memory.chat_memory.add_ai_message(error_msg)
        except Exception as exc:
            error_msg = f"Error: {exc}. Please try again or wait a moment."
            print(f"\n⚠️  {error_msg}")
            memory.chat_memory.add_ai_message(error_msg)

# ------------------------------------------------------------------- #
# ENTRY POINT
# ------------------------------------------------------------------- #
if __name__ == "__main__":
    # Install required packages if not present
    try:
        from langchain.memory import ConversationSummaryBufferMemory
        from langchain.schema import BaseMessage, HumanMessage, AIMessage
        from langchain.llms.base import LLM
    except ImportError:
        print("Installing required LangChain packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "langchain"])
        from langchain.memory import ConversationSummaryBufferMemory
        from langchain.schema import BaseMessage, HumanMessage, AIMessage
        from langchain.llms.base import LLM
    
    interactive_cli()