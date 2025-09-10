import os
import re
import json
import sqlite3
import faiss
import torch                    
import numpy as np
import pandas as pd
import uuid
from openai import OpenAI
from functools import lru_cache
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from datetime import datetime
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
SESSION_DB_PATH  = "chatbot_sessions.db"  # NEW: Session database
NTEE_PATH        = "NTEE_descriptions.csv"
SAMPLEQUERY_PATH = "sample_query.jsonl"
ROW_THRESHOLD    = 500          
TOP_K_DEFAULT    = 10
MEMORY_WINDOW_SIZE = 4  # Configurable memory window

# -------------------------------------------------------------------------- #
# 2.  CONNECT TO MISTRAL-LARGE                                   
# -------------------------------------------------------------------------- #
client = OpenAI(
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1",   
)

MISTRAL_MODEL_ID = "mistral-large-2411"

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

# Back-compat helper 
llama_generate = mistral_generate     

# -------------------------------------------------------------------------- #
# 3.  SESSION MEMORY MANAGEMENT FUNCTIONS
# -------------------------------------------------------------------------- #

def create_new_session(user_info: str = None, memory_window: int = MEMORY_WINDOW_SIZE) -> str:
    """Create a new session and return the session_id."""
    session_id = str(uuid.uuid4())
    
    conn = sqlite3.connect(SESSION_DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO sessions (session_id, user_info)
            VALUES (?, ?)
        """, (session_id, user_info))
        
        cursor.execute("""
            INSERT INTO session_config (session_id, memory_window_size)
            VALUES (?, ?)
        """, (session_id, memory_window))
        
        conn.commit()
        return session_id
        
    except sqlite3.Error as e:
        print(f"❌ Error creating session: {e}")
        return None
    finally:
        conn.close()

def add_chat_to_session(session_id: str, user_query: str, expanded_query: str, 
                       bot_response: str, sql_executed: str = None, 
                       response_type: str = 'pipeline'):
    """Add a new chat exchange to the session and handle memory window management."""
    conn = sqlite3.connect(SESSION_DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Get current chat count
        cursor.execute("""
            SELECT COALESCE(MAX(chat_index), 0) + 1 
            FROM chat_history 
            WHERE session_id = ?
        """, (session_id,))
        next_index = cursor.fetchone()[0]
        
        # Insert new chat
        cursor.execute("""
            INSERT INTO chat_history 
            (session_id, chat_index, user_query, expanded_query, bot_response, sql_executed, response_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, next_index, user_query, expanded_query, bot_response, sql_executed, response_type))
        
        # Update session last_active
        cursor.execute("""
            UPDATE sessions 
            SET last_active = CURRENT_TIMESTAMP 
            WHERE session_id = ?
        """, (session_id,))
        
        # Get memory window size
        cursor.execute("""
            SELECT memory_window_size FROM session_config WHERE session_id = ?
        """, (session_id,))
        window_size = cursor.fetchone()[0]
        
        # Check if we need to summarize old chats
        if next_index > window_size:
            _summarize_old_chats(cursor, session_id, window_size)
        
        conn.commit()
        
    except sqlite3.Error as e:
        print(f"❌ Error adding chat: {e}")
        conn.rollback()
    finally:
        conn.close()

def _summarize_old_chats(cursor, session_id: str, window_size: int):
    """Summarize chats that fall outside the memory window."""
    # Find the oldest chat that needs summarizing
    cursor.execute("""
        SELECT chat_index, user_query, bot_response 
        FROM chat_history 
        WHERE session_id = ? AND chat_index = (
            SELECT MAX(chat_index) - ? FROM chat_history WHERE session_id = ?
        )
    """, (session_id, window_size, session_id))
    
    chat_to_summarize = cursor.fetchone()
    
    if chat_to_summarize:
        chat_idx, user_q, bot_r = chat_to_summarize
        
        # Create summary using Mistral
        summary_prompt = f"""
        Summarize this chat exchange in 1-2 sentences, focusing on the key question and main findings:
        
        User: {user_q}
        Bot: {bot_r}
        
        Summary:"""
        
        try:
            summary = mistral_generate(summary_prompt, max_new=100)
        except:
            summary = f"User asked about: {user_q[:50]}... Bot provided grant/foundation data."
        
        # Insert summary
        cursor.execute("""
            INSERT INTO chat_summaries 
            (session_id, summary_text, covers_chat_start, covers_chat_end)
            VALUES (?, ?, ?, ?)
        """, (session_id, summary, chat_idx, chat_idx))

def get_session_context(session_id: str) -> dict:
    """Retrieve complete context for a session (summary + recent chats)."""
    conn = sqlite3.connect(SESSION_DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Get session config
        cursor.execute("""
            SELECT memory_window_size FROM session_config WHERE session_id = ?
        """, (session_id,))
        config_result = cursor.fetchone()
        window_size = config_result[0] if config_result else MEMORY_WINDOW_SIZE
        
        # Get recent chats (within memory window)
        cursor.execute("""
            SELECT chat_index, user_query, expanded_query, bot_response, sql_executed, timestamp
            FROM chat_history 
            WHERE session_id = ?
            ORDER BY chat_index DESC
            LIMIT ?
        """, (session_id, window_size))
        recent_chats = list(reversed(cursor.fetchall()))  # Oldest first
        
        # Get summaries of older chats
        cursor.execute("""
            SELECT summary_text, covers_chat_start, covers_chat_end
            FROM chat_summaries 
            WHERE session_id = ?
            ORDER BY covers_chat_end ASC
        """, (session_id,))
        summaries = cursor.fetchall()
        
        return {
            'session_id': session_id,
            'window_size': window_size,
            'recent_chats': recent_chats,
            'chat_summaries': summaries,
            'total_chats': len(recent_chats)
        }
        
    except sqlite3.Error as e:
        print(f"Error retrieving session context: {e}")
        return {'session_id': session_id, 'recent_chats': [], 'chat_summaries': []}
    finally:
        conn.close()

def context_aware_reframe_with_llm(user_query: str, session_id: str) -> str:
    """
    Use LLM and conversation context to reframe broad or contextual queries.
    Returns the original query if no reframing is needed.
    """
    # Get session context
    context = get_session_context(session_id)
    
    # Build conversation context - focus on recent relevant exchanges
    context_parts = []
    
    # Add chat summaries (older context) - but limit to recent summaries
    for summary_text, start_idx, end_idx in context['chat_summaries'][-2:]:  # Only last 2 summaries
        context_parts.append(f"Earlier conversation: {summary_text}")
    
    # Add recent detailed chats - but focus on the immediate context
    for chat_idx, user_q, expanded_q, bot_r, sql_exec, timestamp in context['recent_chats'][-2:]:  # Only last 2 chats
        context_parts.append(f"User asked: {user_q}")
        # Truncate bot response to focus on key info
        bot_summary = bot_r[:200] + "..." if len(bot_r) > 200 else bot_r
        context_parts.append(f"Assistant responded: {bot_summary}")
    
    conversation_context = "\n".join(context_parts) if context_parts else "No previous conversation."
    
    # Enhanced reframing prompt with better instructions
    reframing_prompt = f"""
You are helping reframe a user's follow-up question to make it complete and standalone for a database query.

RECENT CONVERSATION CONTEXT:
{conversation_context}

CURRENT USER INPUT: "{user_query}"

TASK: Transform this follow-up question into a complete, standalone question that:
1. Incorporates the relevant context from the previous conversation
2. Is specific and clear about what is being asked
3. Can be understood without needing the previous conversation
4. Maintains the exact same intent as the original follow-up

CRITICAL: Pay close attention to:
- What entity/subject was being discussed (e.g., "Gates Foundation")
- What type of question was asked (e.g., "first organization", "second organization")
- The logical progression of the conversation

EXAMPLES:

Previous: "Which was the first organization that the Gates Foundation gave a grant to?"
Current: "and the 2nd?"
→ "Which was the second organization that the Gates Foundation gave a grant to?"

Previous: "List foundations that support education in California"
Current: "what about healthcare?"
→ "List foundations that support healthcare in California"

Previous: "How many organizations received grants over $100000 in 2020?"
Current: "list them"
→ "List all organizations that received grants over $100000 in 2020"

Previous: "Top 5 grantees in New York"
Current: "in Texas?"
→ "Who are the top 5 grantees in Texas?"

Previous: "Foundations with assets over $1 billion"
Current: "smaller ones?"
→ "List foundations with assets under $1 billion"

IMPORTANT: 
- Keep the same question type (if asking for "second", keep it as "second", not "top 2")
- Preserve specific entities mentioned (Gates Foundation, specific states, etc.)
- Don't change the core intent or add unnecessary limitations

Return ONLY the complete reformulated question:
"""

    try:
        reframed = mistral_generate(reframing_prompt, max_new=200)
        reframed = reframed.strip().strip('"').strip("'")
        
        # Additional validation: if the reformulation seems too different or generic, try fallback
        if len(reframed) < 10 or reframed.lower() == user_query.lower():
            print(f"LLM reformulation seems inadequate: '{reframed}', trying enhanced fallback...")
            reframed = _enhanced_fallback_reformulation(user_query, context)
        
        print(f"Context-aware reframing: '{user_query}' → '{reframed}'")
        return reframed
        
    except Exception as e:
        print(f"LLM reframing failed: {e}")
        return _enhanced_fallback_reformulation(user_query, context)

def _enhanced_fallback_reformulation(user_query: str, context: dict) -> str:
    """
    Enhanced fallback logic for reformulation when LLM fails
    """
    if not context['recent_chats']:
        return user_query
    
    # Get the most recent user question and response
    last_chat = context['recent_chats'][-1]
    last_user_query = last_chat[1]
    last_bot_response = last_chat[3]
    
    query_lower = user_query.lower().strip()
    
    # Handle ordinal follow-ups (2nd, 3rd, etc.)
    ordinal_patterns = [
        (r'\b(and\s+)?the\s*2nd\??', 'second'),
        (r'\b(and\s+)?2nd\??', 'second'), 
        (r'\b(and\s+)?the\s*3rd\??', 'third'),
        (r'\b(and\s+)?3rd\??', 'third'),
        (r'\b(and\s+)?the\s*4th\??', 'fourth'),
        (r'\b(and\s+)?4th\??', 'fourth'),
    ]
    
    for pattern, ordinal_word in ordinal_patterns:
        if re.search(pattern, query_lower):
            # Extract key elements from previous question
            if "first" in last_user_query.lower():
                # Replace "first" with the ordinal
                reformulated = re.sub(r'\bfirst\b', ordinal_word, last_user_query, flags=re.I)
                return reformulated
            elif "1st" in last_user_query.lower():
                reformulated = re.sub(r'\b1st\b', ordinal_word, last_user_query, flags=re.I)
                return reformulated
    
    # Handle "what about X" patterns
    what_about_match = re.search(r'what about\s+(.+?)(?:\?|$)', query_lower)
    if what_about_match:
        new_topic = what_about_match.group(1).strip()
        # Try to substitute the topic in the previous question
        # This is a simple heuristic - could be improved
        if any(word in last_user_query.lower() for word in ['education', 'healthcare', 'environment', 'arts']):
            # Replace the topic
            for old_topic in ['education', 'healthcare', 'environment', 'arts', 'animal', 'technology']:
                if old_topic in last_user_query.lower():
                    reformulated = re.sub(old_topic, new_topic, last_user_query, flags=re.I)
                    return reformulated
        return f"List foundations that support {new_topic}"
    
    # Handle "list them" patterns
    if any(phrase in query_lower for phrase in ["list them", "show them", "list all"]):
        if "how many" in last_user_query.lower():
            # Convert "how many" to "list"
            reformulated = re.sub(r'how many\s+', 'List all ', last_user_query, flags=re.I)
            reformulated = re.sub(r'\?.*$', '', reformulated)  # Remove question mark and everything after
            return reformulated + "?"
    
    # Generic fallback
    return f"Continue the previous question about: {user_query}"

def list_active_sessions(limit: int = 5):
    """Show recent active sessions."""
    conn = sqlite3.connect(SESSION_DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT session_id, created_at, last_active, 
                   (SELECT COUNT(*) FROM chat_history WHERE session_id = s.session_id) as chat_count
            FROM sessions s
            WHERE status = 'active'
            ORDER BY last_active DESC
            LIMIT ?
        """, (limit,))
        
        sessions = cursor.fetchall()
        if sessions:
            print(f"\nRecent Sessions:")
            for i, (sid, created, last_active, chat_count) in enumerate(sessions, 1):
                short_id = sid[:8]
                print(f"  {i}. {short_id}... | {chat_count} chats | Last: {last_active}")
        return sessions
        
    except sqlite3.Error as e:
        print(f"Error listing sessions: {e}")
        return []
    finally:
        conn.close()

def select_or_create_session() -> str:
    """Allow user to resume existing session or create new one."""
    sessions = list_active_sessions(5)
    
    if sessions:
        print("  n. Create new session")
        choice = input("\nSelect session number (1-5) or 'n' for new: ").strip().lower()
        
        if choice == 'n':
            return create_new_session()
        elif choice.isdigit() and 1 <= int(choice) <= len(sessions):
            selected_session = sessions[int(choice) - 1]
            print(f"Resuming session: {selected_session[0][:8]}...")
            return selected_session[0]
    
    # Default: create new session
    print("Creating new session...")
    return create_new_session()

# -------------------------------------------------------------------------- #
# 4.  EXISTING HELPER FUNCTIONS (UNCHANGED)
# -------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load 'all-MiniLM-L6-v2' once and cache it."""
    print("Loading MiniLM embedder...")
    return SentenceTransformer("all-MiniLM-L6-v2")

EMBEDDER = get_embedder()

# Initialize database connection check
with sqlite3.connect(DB_PATH) as conn:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    ALL_TABLES = [row[0] for row in cur.fetchall()]

print("Core initialisation complete. Tables detected:", ", ".join(ALL_TABLES))

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
    "user_data":          "RRNA_Org_Data",
    "organization_data":  "RRNA_Org_Data",
    "organizations":      "RRNA_Org_Data",
    "orgs":               "RRNA_Org_Data",
    "grants":             "RRNAResearchGrants",
    "research_grants":    "RRNAResearchGrants",
}

def replace_table_aliases(sql: str) -> str:
    """Swap out any alias in TABLE_MAPPING with the real table."""
    for alias, real in TABLE_MAPPING.items():
        sql = re.sub(rf"\b{alias}\b", real, sql, flags=re.I)
    return sql

DEBUG_SQL = False       

def _dbg(label: str, msg: str):
    """Lightweight debug printer controlled by DEBUG_SQL flag."""
    if DEBUG_SQL:
        print(f"\n{label}: {msg}\n")

def get_ntee_code_from_csv(user_query: str, csv_path: str = NTEE_PATH):
    """Semantic search over NTEE descriptions."""
    df = pd.read_csv(csv_path, encoding="latin1")

    required = {"org_class_code", "class_short_description"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain 'org_class_code' and 'class_short_description'.")

    desc_embeddings = EMBEDDER.encode(df["class_short_description"].tolist())
    user_embedding  = EMBEDDER.encode([user_query])

    desc_norm  = desc_embeddings / np.linalg.norm(desc_embeddings, axis=1, keepdims=True)
    query_norm = user_embedding / np.linalg.norm(user_embedding)
    sims       = np.dot(desc_norm, query_norm.T).squeeze()

    best_idx   = int(np.argmax(sims))
    return (
        df.iloc[best_idx]["org_class_code"],
        df.iloc[best_idx]["class_short_description"],
    )

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee legacy column names for compatibility."""
    rename_map = {
        "question":   "Question",
        "Question":   "Question",
        "sql_query":  "SQL Query",
        "SQL Query":  "SQL Query",
        "sql":        "SQL Query",
    }
    return df.rename(columns=rename_map, errors="ignore")

def classify_query(query: str, positive_examples: list[str], negative_examples: list[str]) -> bool:
    """Returns True if the query is closer to the positive centroid than the negative."""
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

def fetch_schema_from_db(db_path: str = DB_PATH) -> dict[str, list[str]]:
    """Return mapped schema information."""
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

def generate_multiple_sql_queries_with_schema(user_query: str, schemas: dict[str, list[str]]):
    """Generate SQL queries using schema + few-shot examples + NTEE enrichment."""
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

Generate the most relevant SQLite-compatible SQL query for the given prompt. Ensure that the
query is syntactically correct and works with the provided schema.
If you have to compare something, use the LIKE operator. For state codes,
use abbreviations. Do NOT use the following columns in any query:
{RESTRICTED_COLUMNS}.
Return ONLY the SQL query in a single line. Do not include any explanations.
""".strip()

    llm_output = llama_generate(prompt)
    sql_lines = [ln.strip() for ln in llm_output.splitlines() if ln.strip()]
    return sql_lines[:3]

BROAD_KEYWORDS = (
    " all ", " every ", " list of all ", " show all ", " give me all "
)

def is_broad_query(q: str) -> bool:
    """Check if query is overly broad."""
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
    """Local rewrite: 'all organisations …' → 'top-K organisations …'."""
    out = re.sub(r"\blist of all\b", f"top {k}", q, flags=re.I)
    out = re.sub(r"\ball\b", f"top {k}", out, flags=re.I)
    if out == q:
        out = f"Show the top {k} results for: {q}"
    return out

def _count_rows(conn: sqlite3.Connection, sql: str) -> int:
    """Estimate result size safely."""
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

def execute_sql_query(sql_query: str, db_path: str = DB_PATH):
    """Run SQL query and return rows or error string."""
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
    """Remove ``` fences and keep only the first SQL statement."""
    q = re.sub(r"`{3}sql|`", "", query).strip()
    if ";" in q:
        q = q.split(";", 1)[0]
    return q.strip()

def summarize_findings_with_llm(user_query: str, schema: dict[str, list[str]], results) -> str:
    """Produce a ~50-word explanation of results."""
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

def save_query_sql_pair(question: str, sql: str, jsonl_path: str = SAMPLEQUERY_PATH) -> None:
    """Append a (Q, SQL) pair to sample_query.jsonl if not already there."""
    if os.path.exists(jsonl_path):
        df = pd.read_json(jsonl_path, lines=True)
        df = _ensure_cols(df)
    else:
        df = pd.DataFrame(columns=["Question", "SQL Query"])

    duplicate = ((df["Question"] == question) & (df["SQL Query"] == sql)).any()
    if duplicate:
        return

    with open(jsonl_path, "a", encoding="utf-8") as f:
        json.dump({"question": question, "sql_query": sql}, f, ensure_ascii=False)
        f.write("\n")

    _load_sample_queries.cache_clear()

def run_pipeline(user_query: str, db_path: str = DB_PATH):
    """End-to-end pipeline: schema → SQL → execution → enrichment → summary."""
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
                print(f"Suggestion → {alt_prompt}")

                ans = input("Use this reframed question? [y/N]: ").strip().lower()
                if ans == "y":
                    alt_sqls = generate_multiple_sql_queries_with_schema(alt_prompt, schemas)
                    if alt_sqls:
                        cleaned   = clean_sql_query(alt_sqls[0])
                        raw_sql   = revert_aliases_in_query(cleaned)
                        user_query = alt_prompt
                        print("… proceeding with reframed query.\n")
                else:
                    print("… keeping the original request.\n")
           
            _dbg("CLEANED SQL", raw_sql)
            results = execute_sql_query(raw_sql, db_path)

            if results and not isinstance(results, str):
                try:
                    df_ntee   = pd.read_csv(NTEE_PATH, encoding="latin1")
                    code2desc = dict(zip(df_ntee["org_class_code"], df_ntee["class_short_description"]))

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
    """Load sample_query.jsonl → DataFrame → FAISS index."""
    if not os.path.exists(SAMPLEQUERY_PATH):
        df = pd.DataFrame(columns=["Question", "SQL Query"])
        return df, None, None

    df = pd.read_json(SAMPLEQUERY_PATH, lines=True)
    df = _ensure_cols(df)

    questions = df["Question"].tolist()
    if not questions:
        return df, questions, None

    embeds = EMBEDDER.encode(questions).astype("float32")

    index = faiss.IndexFlatIP(embeds.shape[1])
    faiss.normalize_L2(embeds)
    index.add(embeds)

    return df, questions, index

def get_top_similar_queries(user_query: str, k: int = 3):
    """Find top-k similar queries from the question bank."""
    df, questions, index = _load_sample_queries()
    if index is None:
        return [], df
    
    q_embed = EMBEDDER.encode([user_query]).astype("float32")
    faiss.normalize_L2(q_embed)
    dist, idx = index.search(q_embed, k)
    tops = [(questions[i], float(d), i) for d, i in zip(dist[0], idx[0]) if i < len(questions)]
    return tops, df

def is_followup_question_llm(user_query: str, session_id: str) -> tuple[bool, str]:
    """
    Use LLM to intelligently detect if a query is a follow-up question.
    Returns (is_followup, reasoning)
    """
    context = get_session_context(session_id)
    
    # If no conversation history, it can't be a follow-up
    if not context['recent_chats'] and not context['chat_summaries']:
        return False, "No conversation history"
    
    # Build recent conversation context (last 2-3 exchanges)
    recent_context = []
    
    # Add most recent chats (limit to last 2-3 for focus)
    for chat_idx, user_q, expanded_q, bot_r, sql_exec, timestamp in context['recent_chats'][-3:]:
        recent_context.append(f"User: {user_q}")
        recent_context.append(f"Assistant: {bot_r[:150]}...")  # Truncate long responses
    
    conversation_context = "\n".join(recent_context) if recent_context else "No recent conversation."
    
    classification_prompt = f"""
You are analyzing whether a user's new question is a follow-up to previous conversation or a completely new topic.

RECENT CONVERSATION:
{conversation_context}

NEW USER QUESTION: "{user_query}"

TASK: Determine if this new question is:
1. A FOLLOW-UP: References, continues, or builds upon the previous conversation
2. A NEW TOPIC: Completely independent topic/question

FOLLOW-UP INDICATORS:
- References previous results ("a few more", "show me others", "what about X instead")
- Asks for variations ("similar ones", "in different states", "larger amounts")
- Continues the same domain/topic with modifications
- Uses contextual pronouns ("them", "those", "it", "that")
- Asks for comparisons to previous results
- Requests different filters on same data

NEW TOPIC INDICATORS:
- Completely different subject matter
- No reference to previous conversation
- Standalone question that makes sense without context
- Different data domain entirely

EXAMPLES:
Previous: "foundations supporting animal welfare"
New: "a few more?" → FOLLOW-UP (clearly referencing previous results)

Previous: "healthcare foundations in California" 
New: "what about education foundations?" → FOLLOW-UP (same type, different sector)

Previous: "foundations supporting animal welfare"
New: "how do I start a nonprofit?" → NEW TOPIC (completely different subject)

Previous: "top 10 grantees by funding"
New: "weather in New York" → NEW TOPIC (unrelated domain)

RESPOND WITH ONLY:
FOLLOW-UP: [brief reason]
OR
NEW TOPIC: [brief reason]
"""

    try:
        response = mistral_generate(classification_prompt, max_new=100)
        response = response.strip()
        
        if response.startswith("FOLLOW-UP:"):
            reasoning = response[11:].strip()
            return True, reasoning
        elif response.startswith("NEW TOPIC:"):
            reasoning = response[10:].strip()
            return False, reasoning
        else:
            # Fallback parsing
            if "follow-up" in response.lower() or "follow up" in response.lower():
                return True, "LLM indicated follow-up"
            else:
                return False, "LLM indicated new topic"
                
    except Exception as e:
        print(f"LLM classification failed: {e}")
        # Fallback to simple heuristics
        return _fallback_followup_detection(user_query, context)

def _fallback_followup_detection(user_query: str, context: dict) -> tuple[bool, str]:
    """
    Fallback heuristics if LLM classification fails
    """
    query_lower = user_query.lower().strip()
    
    # Strong follow-up indicators
    strong_indicators = [
        "more", "others", "additional", "also", "too", 
        "what about", "how about", "similar", "like that",
        "them", "those", "these", "it", "that",
        "instead", "rather", "alternatively"
    ]
    
    # Check for very short queries (likely follow-ups)
    if len(query_lower.split()) <= 3:
        for indicator in strong_indicators:
            if indicator in query_lower:
                return True, f"Short query with '{indicator}'"
    
    # Check for contextual pronouns at start
    if query_lower.startswith(("what about", "how about", "show me", "list", "give me")):
        if any(word in query_lower for word in ["more", "other", "different", "similar"]):
            return True, "Request for variations"
    
    return False, "No clear follow-up indicators"

def enhanced_pipeline_with_smart_followup_detection(user_query: str, session_id: str, db_path: str = DB_PATH):
    """
    Enhanced pipeline that first determines if query is follow-up, then handles appropriately.
    Returns: (sql_executed, results, summary, reformulated_query)
    """
    
    # STEP 1: Intelligent follow-up detection
    is_followup, reasoning = is_followup_question_llm(user_query, session_id)
    
    print(f"Follow-up analysis: {is_followup} - {reasoning}")
    
    reformulated_query = None
    working_query = user_query
    
    # STEP 2: Handle based on classification
    if is_followup:
        print("Detected follow-up question - using context-aware reformulation...")
        reformulated_query = context_aware_reframe_with_llm(user_query, session_id)
        working_query = reformulated_query
        print(f"Reformulated '{user_query}' -> '{working_query}'")
    else:
        print("Detected new topic - processing as standalone query...")
        # For new topics, still check if query is too broad
        broad_indicators = [r'\ball\s+', r'\bevery\s+', r'\blist\s+all\b', r'\bshow\s+all\b']
        is_broad = any(re.search(pattern, user_query, re.I) for pattern in broad_indicators)
        
        if is_broad:
            print("New topic but broad query - applying basic reframing...")
            reformulated_query = context_aware_reframe_with_llm(user_query, session_id)
            working_query = reformulated_query
    
    # STEP 3: Continue with normal pipeline
    schemas = fetch_schema_from_db(db_path)
    candidate_sql = generate_multiple_sql_queries_with_schema(working_query, schemas)

    conn = sqlite3.connect(db_path)      
    try:
        for sql in candidate_sql:
            cleaned = clean_sql_query(sql)
            raw_sql = revert_aliases_in_query(cleaned)
            raw_sql = replace_table_aliases(raw_sql)

            # Check row count and auto-handle for web
            row_cnt = _count_rows(conn, raw_sql)
            if row_cnt > ROW_THRESHOLD:
                print(f"Query would return ≈{row_cnt:,} rows (threshold = {ROW_THRESHOLD}).")
                print("Auto-applying LIMIT 10 for web interface...")
                if "LIMIT" not in raw_sql.upper():
                    if "ORDER BY" in raw_sql.upper():
                        raw_sql = raw_sql + " LIMIT 10"
                    else:
                        raw_sql = raw_sql + " ORDER BY 1 DESC LIMIT 10"
                print("✓ Applied LIMIT 10 to results.\n")
           
            print(f"Executing SQL: {raw_sql}")
            results = execute_sql_query(raw_sql, db_path)

            if results and not isinstance(results, str):
                # NTEE enrichment
                try:
                    df_ntee = pd.read_csv(NTEE_PATH, encoding="latin1")
                    code2desc = dict(zip(df_ntee["org_class_code"], df_ntee["class_short_description"]))

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

                summary = summarize_findings_with_llm(working_query, schemas, results)

                try:
                    question_to_save = reformulated_query if reformulated_query else user_query
                    save_query_sql_pair(question_to_save, cleaned)
                except Exception as exc:
                    print(f"Failed to save query pair: {exc}")

                return cleaned, results, summary, reformulated_query

    finally:
        conn.close()

    return "No valid results found for any query.", None, None, reformulated_query

#  -------------------------------------------------------------------------- #
# 5. SIMPLIFIED INTERACTIVE CLI WITH SMART REFORMULATION
# -------------------------------------------------------------------------- #

def interactive_cli():
    """Main CLI with session-based memory and smart reformulation."""
    print("─" * 60)
    print("  CharityBot — ask about grants, foundations or sectors")
    print("  (type 'exit' to quit, 'memory' to see conversation history)")
    print("─" * 60)

    current_session_id = select_or_create_session()
    
    if not current_session_id:
        print("Failed to create/select session. Exiting...")
        return
        
    print(f"Session active: {current_session_id[:8]}...\n")

    while True:
        try:
            user_q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_q.lower() in {"exit", "quit"}:
            break
        
        if user_q.lower() == "memory":
            context = get_session_context(current_session_id)
            print(f"\nSession Memory ({current_session_id[:8]}...):")
            
            if context['chat_summaries']:
                print("Earlier conversation summaries:")
                for summary_text, start_idx, end_idx in context['chat_summaries']:
                    print(f"    • Chats {start_idx}-{end_idx}: {summary_text}")
            
            if context['recent_chats']:
                print(f"Recent chats (last {context['window_size']}):")
                for chat_idx, user_q_mem, expanded_q, bot_r, sql_exec, timestamp in context['recent_chats']:
                    print(f"    {chat_idx}. You: {user_q_mem}")
                    print(f"       Bot: {bot_r[:80]}{'...' if len(bot_r) > 80 else ''}")
            continue
        
        if user_q.lower() == "session":
            print(f"\nCurrent session: {current_session_id}")
            new_session = select_or_create_session()
            if new_session:
                current_session_id = new_session
                print(f"Switched to session: {current_session_id[:8]}...")
            continue

        # Try similarity search first
        tops, df = get_top_similar_queries(user_q)
        
        if tops:
            print("\nDid you mean:")
            for i, (q, score, _) in enumerate(tops, 1):
                print(f"  {i}. {q}    [similarity {score:.2f}]")
            print("  0. None of the above")

            choice = input("Select 0-3 and press Enter: ").strip()

            if choice.isdigit() and 1 <= int(choice) <= len(tops):
                _, _, idx = tops[int(choice) - 1]
                canned_sql = df.iloc[idx]["SQL Query"]
                schemas = fetch_schema_from_db(DB_PATH)
                cleaned = clean_sql_query(canned_sql)
                
                results = execute_sql_query(cleaned, DB_PATH)
                summary = summarize_findings_with_llm(user_q, schemas, results)
                
                print("\nBot:", summary)
                add_chat_to_session(current_session_id, user_q, user_q, summary, cleaned, 'cached')
                continue

        # Handle broad queries
        if is_broad_query(user_q):
            alt_q = quick_reframe(user_q, TOP_K_DEFAULT)
            print(f"\nThat request is likely to return too many rows.\nHow about: {alt_q}")
            ans = input("1 = Yes, proceed | 0 = No, I'll rephrase: ").strip()
            if ans == "1":
                user_q = alt_q        
            else:
                print("Okay, please try a different question.")
                continue                 
        
        try:
            # Use enhanced pipeline with smart reformulation
            sql_executed, _, summary = enhanced_pipeline_with_smart_followup_detection(user_q, current_session_id, DB_PATH)
            
            if summary:
                print("\nBot:", summary)
                add_chat_to_session(current_session_id, user_q, user_q, summary, sql_executed, 'pipeline')
            else:
                error_msg = "I couldn't find relevant results for that query. Please try rephrasing."
                print(f"\nBot: {error_msg}")
                add_chat_to_session(current_session_id, user_q, user_q, error_msg, None, 'error')
                
        except Exception as exc:
            error_msg = f"Error: {exc}. Please try again or wait a moment."
            print(f"\n{error_msg}")
            add_chat_to_session(current_session_id, user_q, user_q, error_msg, None, 'error')


# -------------------------------------------------------------------------- #
# 6. ENTRY POINT
# -------------------------------------------------------------------------- #

def check_session_database():
    """Check if session database exists, if not, guide user to create it."""
    if not os.path.exists(SESSION_DB_PATH):
        print(f" Session database '{SESSION_DB_PATH}' not found!")
        print("Please run 'database.py' first to create the session database.")
        print("Example: python database.py")
        return False
    
    # Test connection
    try:
        conn = sqlite3.connect(SESSION_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions';")
        if not cursor.fetchone():
            print(" Session tables not found in database!")
            print("Please run 'database.py' to create the required tables.")
            return False
        conn.close()
        return True
    except Exception as e:
        print(f" Error accessing session database: {e}")
        return False

if __name__ == "__main__":
    print("🏗️ Starting CharityBot with Smart Context Reformulation...")
    
    if not check_session_database():
        exit(1)
    
    print("Session database ready!")
    
    if not os.path.exists(DB_PATH):
        print(" Main database not found!")
        exit(1)
    
    try:
        interactive_cli()  # Use the single clean function
    except KeyboardInterrupt:
        print("\n\nCharityBot session ended. Goodbye!")
    except Exception as e:
        print(f"\nUnexpected error: {e}")