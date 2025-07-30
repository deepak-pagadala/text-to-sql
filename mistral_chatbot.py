import os
import re
import sqlite3
import faiss
import torch                    
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from functools import lru_cache
from dotenv import load_dotenv

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
SAMPLEQUERY_PATH = "sample_query.csv"
ROW_THRESHOLD = 500          
TOP_K_DEFAULT = 10

# -------------------------------------------------------------------------- #
# 2.  CONNECT TO MISTRAL-LARGE                                    
# -------------------------------------------------------------------------- #
client = OpenAI(
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1",   
)

MISTRAL_MODEL_ID = "mistral-large-latest"

def mistral_generate(prompt: str, max_new: int = 256) -> str:
    """
    Call Mistral Large and return the assistant’s reply as plain text.
    """
    response = client.chat.completions.create(
        model=MISTRAL_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_new,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

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


"""
Column-renaming map used to translate verbose Llama SQL aliases ⇆ actual DB
column names, plus a small list of columns we never expose to the user.
"""

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
        print(f"\n🔎  {label}: {msg}\n")

# ------------------------------------------------------------------- #
#   NTEE helper
# ------------------------------------------------------------------- #
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

# ------------------------------------------------------------------- #
#  Query-type classifiers (foundation / grantee / sector)
# ------------------------------------------------------------------- #
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


# ------------------------------------------------------------------- #
# 4.  FETCH DB SCHEMA
# ------------------------------------------------------------------- #
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


# ------------------------------------------------------------------- #
#  BUILD A FAISS INDEX OVER SCHEMA TEXT
# ------------------------------------------------------------------- #
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


# ------------------------------------------------------------------- #
#   GENERATE UP TO 3 SQL QUERIES
# ------------------------------------------------------------------- #
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
# ------------------------------------------------------------------- #
#  BASIC BROAD-QUERY HEURISTICS   
# ------------------------------------------------------------------- #
TOP_K_DEFAULT  = 10                      
BROAD_KEYWORDS = (
    " all ", " every ", " list of all ", " show all ", " give me all "
)
def is_broad_query(q: str) -> bool:
    """
    Heuristic:
      • must contain an 'all/every/…' keyword   AND
      • must NOT contain an explicit row-limit hint
        – keywords  (top, first, limit …) OR
        – a small integer (≤ 50)
    Years like 2024 are *ignored*; they don’t count as limits.
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
    Local (no-LLM) rewrite: ‘all organisations …’ → ‘top-K organisations …’.
    """
    out = re.sub(r"\blist of all\b", f"top {k}", q, flags=re.I)
    out = re.sub(r"\ball\b",        f"top {k}", out, flags=re.I)
    if out == q:                                    # nothing matched
        out = f"Show the top {k} results for: {q}"
    return out

# -------------------------------------------------------------------------- #
#  QUESTION REFRAMING PIPELINE                                               #
# -------------------------------------------------------------------------- #


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

# ------------------------------------------------------------------- #
#   EXECUTE SQL AGAINST SQLITE
# ------------------------------------------------------------------- #
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

# ------------------------------------------------------------------- #
#  REVERT PRETTY ALIASES BACK TO REAL COLUMN NAMES
# ------------------------------------------------------------------- #
def revert_aliases_in_query(sql_query: str) -> str:
    """Replace human-friendly column names with their DB originals."""
    reverse = {v: k for k, v in COLUMN_MAPPING.items()}
    for alias, original in reverse.items():
        sql_query = re.sub(rf"\b{alias}\b", original, sql_query)
    return sql_query

# ------------------------------------------------------------------- #
#   STRIP MARKDOWN FENCES     
# ------------------------------------------------------------------- #
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


# ------------------------------------------------------------------- #
#  SUMMARIZE RESULT SET 
# ------------------------------------------------------------------- #
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

# ------------------------------------------------------------------- #
#  APPEND 
# ------------------------------------------------------------------- #
def save_query_sql_pair(question: str, sql: str,
                        csv_path: str = SAMPLEQUERY_PATH) -> None:
    """
    Append the pair only if it doesn’t already exist, then clear the
    LRU-cached FAISS index so the new example is live.
    """
    # load existing
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Question", "SQL Query"])

    # deduplicate on BOTH columns
    duplicate = ((df["Question"] == question) &
                 (df["SQL Query"] == sql)).any()

    if not duplicate:
        df.loc[len(df)] = {"Question": question, "SQL Query": sql}
        df.to_csv(csv_path, index=False)

        # refresh cached index so next call sees the new row
        _load_sample_queries.cache_clear()

# ------------------------------------------------------------------- #
# PIPELINE DRIVER
# ------------------------------------------------------------------- #
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

# ------------------------------------------------------------------- #
# SIMILAR-QUESTION RETRIEVER
# ------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _load_sample_queries():
    df = pd.read_csv(SAMPLEQUERY_PATH)
    questions = df["Question"].tolist()
    embeds = EMBEDDER.encode(questions).astype("float32")
    index = faiss.IndexFlatL2(embeds.shape[1])
    index.add(embeds)
    return df, questions, index


def get_top_similar_queries(user_query: str, k: int = 3):
    df, questions, index = _load_sample_queries()
    q_embed = EMBEDDER.encode([user_query]).astype("float32")
    dist, idx = index.search(q_embed, k)
    tops = [(questions[i], 1 / (1 + d), i) for d, i in zip(dist[0], idx[0])]
    return tops, df

# ------------------------------------------------------------------- #
#  INTERACTIVE CLI                   
# ------------------------------------------------------------------- #
def interactive_cli():
    """
    1. Offer similarity suggestions.
    2. If the user rejects them (chooses 0), decide if the question is
       “too broad” (likely to explode rows/tokens).
    3. If broad → offer a local ‘top-K’ rewrite and ask confirmation.
    4. Only when the user accepts something bounded do we call run_pipeline(),
       which may invoke Mistral.
    """
    print("────────────────────────────────────────────────────────────")
    print("  CharityBot – ask about grants, foundations or sectors")
    print("  (type 'exit' to quit)")
    print("────────────────────────────────────────────────────────────")

    while True:
        try:
            user_q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  Bye!")
            break

        if user_q.lower() in {"exit", "quit"}:
            break

        tops, df = get_top_similar_queries(user_q)
        print("\nDid you mean:")
        for i, (q, score, _) in enumerate(tops, 1):
            print(f"  {i}. {q}    [similarity {score:.2f}]")
        print("  0. None of the above")

        choice = input("Select 0-3 and press Enter: ").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(tops):
            _, _, idx   = tops[int(choice) - 1]
            canned_sql  = df.iloc[idx]["SQL Query"]
            schemas     = fetch_schema_from_db(DB_PATH)
            cleaned     = clean_sql_query(canned_sql)
            _dbg("CANNED SQL", cleaned)

            results     = execute_sql_query(cleaned, DB_PATH)
            summary     = summarize_findings_with_llm(user_q, schemas, results)
            
            print("\nBot:", summary)
            continue

        if is_broad_query(user_q):
            alt_q = quick_reframe(user_q, TOP_K_DEFAULT)
            print(
                "\nThat request is likely to return too many rows."
                f"\nHow about:\n   ► {alt_q}"
                "\n1 = Yes, proceed  \n0 = No, I'll rephrase"
            )
            ans = input("Select 1 or 0 and press Enter: ").strip()
            if ans == "1":
                user_q = alt_q        
            else:
                print("Okay, please try a different question.")
                continue                 
        
        try:
            _, _, summary = run_pipeline(user_q, DB_PATH)
            print("\nBot:", summary)
        except Exception as exc:
            print("\n⚠️  Error:", exc)
            print("   Please try again or wait a moment.")


# ------------------------------------------------------------------- #
# ENTRY POINT
# ------------------------------------------------------------------- #
if __name__ == "__main__":
    interactive_cli()