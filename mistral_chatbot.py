import os
import re
import sqlite3
import faiss
import torch                     # still used for Mini-LM embeddings
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# NEW – Mistral’s OpenAI-compatible client
from openai import OpenAI

from functools import lru_cache
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"  
# -------------------------------------------------------------------------- #
# 1.  ENVIRONMENT VARIABLES & PATHS
# -------------------------------------------------------------------------- #
load_dotenv()                     # reads .env in current working dir

HF_TOKEN        = os.getenv("HF_TOKEN")        # ← still needed by MiniLM
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") # ← NEW

if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY not set in .env")

DB_PATH          = "chatbot22.db"
NTEE_PATH        = "NTEE_descriptions.csv"
SAMPLEQUERY_PATH = "sample_query.csv"


# -------------------------------------------------------------------------- #
# 2.  CONNECT TO MISTRAL-LARGE (hosted)                                      #
# -------------------------------------------------------------------------- #
client = OpenAI(
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1",   # Mistral’s endpoint
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
# 2b.  Back-compat helper (keeps the rest of the file working as-is)
# -------------------------------------------------------------------------- #
llama_generate = mistral_generate      # ← delete later if you rename calls



# -------------------------------------------------------------------------- #
# 3.  OPEN SQLITE DB & LIST TABLES  (runs once at start-up)
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

# convenience var so the rest of the file reads naturally
EMBEDDER = get_embedder()


# ------------------------------------------------------------------- #
# 2.  NTEE helper
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
# 3.  Query-type classifiers (foundation / grantee / sector)
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
# 1.  FETCH DB SCHEMA
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
# 2.  (Optional) BUILD A FAISS INDEX OVER SCHEMA TEXT
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
# 3.  GENERATE UP TO 3 SQL QUERIES WITH Llama-2
# ------------------------------------------------------------------- #
def generate_multiple_sql_queries_with_schema(user_query: str, schemas: dict[str, list[str]]):
    """
    Use the DB schema + few-shot examples + optional NTEE enrichment
    to prompt Llama-2.  Returns a list of ≤ 3 SQL strings.
    """
    # --- a) format schema for the prompt ---------------------------------- #
    schema_block = "\n".join(
        f"Table: {tbl} - Columns: {', '.join(cols)}" for tbl, cols in schemas.items()
    )

    # --- b) NTEE enrichment when needed ----------------------------------- #
    if (is_organization_query(user_query) or is_grantee_query(user_query)) \
       and is_sector_related_query(user_query):
        code, desc = get_ntee_code_from_csv(user_query, NTEE_PATH)
        user_query = f"{user_query} (NTEE Code: {code} - {desc})"

    # --- c) few-shot examples -------------------------------------------- #
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

    # --- d) build the final prompt --------------------------------------- #
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

    # --- e) call Llama-2 --------------------------------------------------- #
    llm_output = llama_generate(prompt)

    # --- f) post-process --------------------------------------------------- #
    sql_lines = [ln.strip() for ln in llm_output.splitlines() if ln.strip()]
    return sql_lines[:3]


# ------------------------------------------------------------------- #
# 1.  EXECUTE SQL AGAINST SQLITE
# ------------------------------------------------------------------- #
def execute_sql_query(sql_query: str, db_path: str = DB_PATH):
    """
    Run `sql_query` on the SQLite DB and return a list of rows.
    On error, returns the SQLite error message as a string.
    """
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
# 2.  REVERT PRETTY ALIASES BACK TO REAL COLUMN NAMES
# ------------------------------------------------------------------- #
def revert_aliases_in_query(sql_query: str) -> str:
    """Replace human-friendly column names with their DB originals."""
    reverse = {v: k for k, v in COLUMN_MAPPING.items()}
    for alias, original in reverse.items():
        sql_query = re.sub(rf"\b{alias}\b", original, sql_query)
    return sql_query


# ------------------------------------------------------------------- #
# 3.  STRIP MARKDOWN FENCES FROM SQL
# ------------------------------------------------------------------- #
def clean_sql_query(query: str) -> str:
    """Remove back-ticks/```sql fences that might wrap the query."""
    return re.sub(r"`{3}sql|`", "", query).strip()


# ------------------------------------------------------------------- #
# 4.  SUMMARISE RESULT SET WITH LLAMA-2
# ------------------------------------------------------------------- #
def summarize_findings_with_llm(
    user_query: str,
    schema: dict[str, list[str]],
    results,
) -> str:
    """
    Produce a ~50-word explanation of `results`, grounded in `schema`.
    """
    # ---- (a) format schema & results ----------------------------------- #
    schema_txt  = "\n".join(
        f"Table: {tbl} - Columns: {', '.join(cols)}" for tbl, cols in schema.items()
    )
    result_txt  = "\n".join(map(str, results))

    # ---- (b) optional sector hint (last column heuristic) -------------- #
    sector_hint = ""
    if results and isinstance(results[0], (list, tuple)) and len(results[0]) >= 3:
        maybe_sector = results[0][-1]
        if isinstance(maybe_sector, str) and len(maybe_sector) > 3:
            sector_hint = f"This query is related to the sector: '{maybe_sector}'.\n\n"

    # ---- (c) build prompt --------------------------------------------- #
    prompt = f"""
User Query: {user_query}

Schema Details:
{schema_txt}

SQL Results:
{result_txt}

{sector_hint}Generate a concise, human-friendly summary of the results in about 50 words.
Be specific about the sector or topic.
""".strip()

    # ---- (d) call Llama-2 --------------------------------------------- #
    return llama_generate(prompt)


# ------------------------------------------------------------------- #
# 1.  PIPELINE DRIVER
# ------------------------------------------------------------------- #
def run_pipeline(user_query: str, db_path: str = DB_PATH):
    """
    End-to-end:   schema → prompt → (≤3) SQL → first query that returns rows →
                  optional NTEE enrichment → 50-word Llama summary
    """
    schemas = fetch_schema_from_db(db_path)
    candidate_sql = generate_multiple_sql_queries_with_schema(user_query, schemas)

    for sql in candidate_sql:
        cleaned = clean_sql_query(sql)
        raw_sql = revert_aliases_in_query(cleaned)
        results = execute_sql_query(raw_sql, db_path)

        if results and not isinstance(results, str):
            # ---------- optional NTEE enrichment ------------------------ #
            try:
                df_ntee = pd.read_csv(NTEE_PATH, encoding="latin1")
                code2desc = dict(
                    zip(df_ntee["org_class_code"], df_ntee["class_short_description"])
                )

                # detect an NTEE-like code in any column
                code_idx = None
                for i, val in enumerate(results[0]):
                    if isinstance(val, str) and re.match(r"^[A-Z]\d{2}", val):
                        if val in code2desc:
                            code_idx = i
                            break

                if code_idx is not None:
                    enriched = []
                    for row in results:
                        row = list(row)
                        row.append(code2desc.get(row[code_idx], "Unknown Sector"))
                        enriched.append(tuple(row))
                    results = enriched
            except Exception:
                pass  # enrichment is best-effort

            summary = summarize_findings_with_llm(user_query, schemas, results)
            return cleaned, results, summary

    return "No valid results found for any query.", None, None


# ------------------------------------------------------------------- #
# 2.  SIMILAR-QUESTION RETRIEVER
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
# 3.  CONSOLE CHAT LOOP  (replaces the Streamlit UI)
# ------------------------------------------------------------------- #
def interactive_cli():
    """
    Simple REPL with suggestion + branching logic that mirrors the Streamlit UI.
    Type 'exit' or press Ctrl-C to quit.
    """
    print("────────────────────────────────────────────────────────────")
    print("  CharityBot – ask about grants, foundations or sectors   ")
    print("  (type 'exit' to quit)                                   ")
    print("────────────────────────────────────────────────────────────")

    while True:
        try:
            user_q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  Bye!")
            break

        if user_q.lower() in {"exit", "quit"}:
            break

        # ------- step A: show similar questions ------------------------ #
        tops, df = get_top_similar_queries(user_q)
        print("\nDid you mean:")
        for i, (q, score, _) in enumerate(tops, 1):
            print(f"  {i}. {q}    [similarity {score:.2f}]")
        print("  0. None of the above")

        choice = input("Select 0-3 and press Enter: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(tops):
            _, _, idx = tops[int(choice) - 1]
            canned_sql = df.iloc[idx]["SQL Query"]
            schemas    = fetch_schema_from_db(DB_PATH)
            cleaned    = clean_sql_query(canned_sql)
            results    = execute_sql_query(cleaned, DB_PATH)
            summary    = summarize_findings_with_llm(user_q, schemas, results)
        else:
            _, _, summary = run_pipeline(user_q, DB_PATH)

        print("\nBot:", summary)


# ------------------------------------------------------------------- #
# 4.  ENTRY POINT
# ------------------------------------------------------------------- #
if __name__ == "__main__":
    interactive_cli()