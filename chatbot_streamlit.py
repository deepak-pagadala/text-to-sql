import os
import sqlite3
import faiss
import numpy as np
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from mistralai import Mistral
import streamlit as st
from dotenv import load_dotenv

# Load secrets from .env file
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
assert api_key is not None, "MISTRAL_API_KEY is missing!"
hf_token = os.getenv("HF_TOKEN")
client = Mistral(api_key)

# paths
db_path = 'chatbot22.db'
ntee_path = 'NTEE_descriptions.csv'
samplequery_path = 'sample_query.csv'  # or the correct path


ntee_descdf = pd.read_csv(ntee_path, encoding='latin1')
st.set_page_config(page_title="CharityBot", page_icon="🤖", layout="centered")

# Columns renaming
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
    "org_assets": "organization_assests",
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
    "giver_id" : "foundation_giver_id",
    "grant_date" : "grant_date",
    "grant_amount" : "grant_amount",
    "g_city" : "grantee_city",
    "g_state" : "grantee_state",
    "g_zip" : "grantee_zipcode",
    "g_ein" : "grantee_ein",
    "g_name" : "grantee_name",
    "f_city" : "foundation_city",
    "f_state" : "foundation_state",
    "f_zip" : "foundation_zipcode",
    "f_state_name" : "foundation_state_name",
    "f_region_4" : "foundation_region_4",
    "f_region_9" : "foundation_region_9",
    "g_state_name" : "grantee_state_name",
    "g_region_4" : "grantee_region_4",
    "g_region_9" : "grantee_region_9",
    "f_ein" : "foundation_ein",
    "f_name" : "foundation_name",
    "grant_source" : "grant_source",
    "formation_yr" : "formation_year",
    "ruling_yr" : "ruling_year",
    "f_mission_partI" : "foundation_mission_partI",
    "f_mission_partIII" : "foundation_mission_partIII",
    "g_mission_partI" : "grantee_mission_partI",
    "g_mission_partIII" : "grantee_mission_partIII",
    "f_ntee_letter" : "foundation_ntee_code",
    "g_ntee_letter" : "grantee_ntee_code",
    "f_ntee_description" : "foundation_ntee_description",
    "g_ntee_description" : "grantee_ntee_description",
    "f_ntee_major8" : "foundation_ntee_short_description",
    "f_ntee_major10" : "foundation_ntee_description_abbreviation",
    "f_ntee_major12" : "foundation_ntee_major12",
    "g_ntee_major8" : "grantee_ntee_short_description",
    "g_ntee_major10" : "grantee_ntee_description_abbreviation",
    "g_ntee_major12" : "grantee_ntee_major12",
    "f_amt_assets_total" : "foundation_assets_total",
    "f_amt_exp_grants" : "foundation_amountt_exp_grants",
    "f_num_employees" : "foundation_number_of_employees",
    "f_num_volunteers" : "foundation_number_of_volunteers",
    "g_amt_assets_total" : "grantee_assets_total",
    "g_amt_exp_total" : "grantee_amount_exp_total",
    "g_num_employees" : "grantee_number_of_employees",
    "g_num_volunteers" : "grantee_number_of_volunteers",
    "g_amt_rev_contrib_total" : "grantee_revenue_contribution_total",
    "g_amt_rev_total" : "grantee_revenue_total"
}

RESTRICTED_COLUMNS = [
    'grantee_ntee_description',
    'foundation_ntee_description',
    'grantee_ntee_major12',
    'grantee_ntee_description_abbreviation',
    'foundation_ntee_major12',
    'foundation_ntee_description_abbreviation',
    'foundation_ntee_short_description',
    'grantee_ntee_short_description',
    'organization_nteecode_first_letter'
]



model = SentenceTransformer('all-MiniLM-L6-v2')

def get_ntee_code_from_csv(user_query, ntee_path):
    """
    Load the CSV file containing NTEE descriptions, perform semantic search over the descriptions,
    and return the best matching NTEE code along with its detailed description.
    """
    df_ntee = pd.read_csv(ntee_path, encoding='latin1')
    if "org_class_code" not in df_ntee.columns or "class_short_description" not in df_ntee.columns:
        raise ValueError("CSV file must contain 'org_class_code' and 'class_short_description' columns.")

    model_ntee = SentenceTransformer('all-MiniLM-L6-v2')
    descriptions = df_ntee["class_short_description"].tolist()
    desc_embeddings = model_ntee.encode(descriptions)
    user_embedding = model_ntee.encode([user_query])

    norm_desc = desc_embeddings / np.linalg.norm(desc_embeddings, axis=1, keepdims=True)
    norm_query = user_embedding / np.linalg.norm(user_embedding)
    cosine_similarities = np.dot(norm_desc, norm_query.T).squeeze()

    best_idx = int(np.argmax(cosine_similarities))
    best_ntee_code = df_ntee.iloc[best_idx]["org_class_code"]
    best_ntee_description = df_ntee.iloc[best_idx]["class_short_description"]

    return best_ntee_code, best_ntee_description


def is_organization_query(query):
    """
    Determines if a query is organization-related.
    Uses semantic similarity with a centroid approach.
    """

    org_examples = [
        "Which foundations have provided the highest grants?",
        "Tell me about the foundations giving grants.",
        "List organizations that offer grants.",
        "What organizations support environmental projects?"
    ]

    non_org_examples = [
        "What is the weather in New York?",
        "Show me the sales numbers for Q3.",
        "How many users signed up today?"
    ]

    return classify_query(query, org_examples, non_org_examples, model)

def is_grantee_query(query):
    """
    Determines if a query is grantee-related.
    """

    grantee_examples = [
        "Which grantees received the highest funding?",
        "Tell me about the grantees that got the most grants.",
        "List grantees who work in education.",
        "Which organizations have received the most grants?"
    ]

    non_grantee_examples = [
        "Which organizations gave the highest grants?",
        "Show foundations supporting healthcare.",
        "How many users signed up today?"
    ]

    return classify_query(query, grantee_examples, non_grantee_examples, model)

def is_sector_related_query(query):
    """
    Determines if the query is related to a sector or industry.
    """

    sector_examples = [
        "Which foundations support education?",
        "Give me organizations in the healthcare sector.",
        "List tech-focused grant providers."
    ]

    non_sector_examples = [
        "Which organizations gave the highest grants?",
        "How many companies donated over $10 million?",
        "List organizations by their total grants."
    ]

    return classify_query(query, sector_examples, non_sector_examples, model)

def classify_query(query, positive_examples, negative_examples, model):
    """
    General function for query classification using centroid-based similarity.
    """
    pos_embeddings = model.encode(positive_examples)
    neg_embeddings = model.encode(negative_examples)
    pos_centroid = np.mean(pos_embeddings, axis=0)
    neg_centroid = np.mean(neg_embeddings, axis=0)

    query_embedding = model.encode([query])[0]
    query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)

    sim_pos = np.dot(query_embedding_norm, pos_centroid / np.linalg.norm(pos_centroid))
    sim_neg = np.dot(query_embedding_norm, neg_centroid / np.linalg.norm(neg_centroid))

    return sim_pos > sim_neg

def fetch_schema_from_db(db_path):
    """
    Extracts the database schema dynamically.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    schemas = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f'PRAGMA table_info("{table_name}");')
        #columns = [column[1] for column in cursor.fetchall()]
        columns = [COLUMN_MAPPING.get(column[1], column[1]) for column in cursor.fetchall()]
        schemas[table_name] = columns

    conn.close()
    return schemas

def create_faiss_index(schema_texts):
    """
    (Optional) Create a FAISS index based on schema text embeddings.
    (Our code: for potential similarity search over schema details)
    """
    #model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(schema_texts)
    embeddings = np.array(embeddings).astype('float32')
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, "index_file.index")
    return faiss_index

def generate_multiple_sql_queries_with_schema(user_query, schemas):
    """
    Generate SQL queries based on the user query, including NTEE enrichment when relevant.
    """

    relevant_schema = "\n".join([f"Table: {table} - Columns: {', '.join(cols)}" for table, cols in schemas.items()])

    if is_organization_query(user_query) and is_sector_related_query(user_query):
          detected_ntee_code, detected_description = get_ntee_code_from_csv(user_query, ntee_path)
          user_query = f"{user_query} (NTEE Code: {detected_ntee_code} - {detected_description})"

    if is_grantee_query(user_query) and is_sector_related_query(user_query):
          detected_ntee_code, detected_description = get_ntee_code_from_csv(user_query, ntee_path)
          user_query = f"{user_query} (NTEE Code: {detected_ntee_code} - {detected_description})"

    # Few-shot examples to guide SQL generation (for structure and syntax)
    few_shot_examples = """
    Example 1:
    Question: Which foundations have given the highest amount of grants?
    SQL Query: SELECT foundation_name, MAX(grant_amount) AS max_grant FROM your_table GROUP BY foundation_name ORDER BY max_grant DESC;

    Example 2:
    Question: Show the top 5 grantees in the healthcare sector.
    SQL Query: SELECT grantee_name, SUM(grant_amount) AS total_grants FROM your_table WHERE sector LIKE '%health%' GROUP BY grantee_name ORDER BY total_grants DESC LIMIT 5;
    """

    prompt = f"""
    User Query: {user_query}

    Few-shot Examples:
    {few_shot_examples}

    Schema Details:
    {relevant_schema}

    Generate the most relevant SQL query for the given prompt. Ensure that the query is syntactically correct and works with the provided schema.
    If you have to compare something, use the LIKE operator. For state codes, use abbreviations.
    Do NOT use the following columns in any query: {RESTRICTED_COLUMNS}.
    Return ONLY the SQL query in a single line. Do not include any explanations.
    """

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    sql_queries = response.choices[0].message.content.strip().splitlines()
    return sql_queries[:3]

def execute_sql_query(db_path, sql_query):
    """
    Execute the given SQL query on the SQLite database and return the results.
    (Our code)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        return results
    except sqlite3.Error as e:
        return f"An error occurred: {e}"
    finally:
        conn.close()

def revert_aliases_in_query(sql_query):
    """
    Convert user-friendly column names back to their original names for execution.
    """
    reverse_mapping = {v: k for k, v in COLUMN_MAPPING.items()}
    for alias, original in reverse_mapping.items():
        sql_query = re.sub(rf"\b{alias}\b", original, sql_query)
    return sql_query

def clean_sql_query(query):
    """
    Clean the SQL query by removing unwanted markdown or formatting.
    (Our code)
    """
    query_cleaned = re.sub(r'`{3}sql|`', '', query).strip()
    return query_cleaned


def summarize_findings_with_llm(query, schema, results):
    """
    Generate a human-friendly summary of the SQL query results using the LLM.
    (Combined approach: our straightforward summarization with LLM prompt techniques)
    """
    schema_text = "\n".join([f"Table: {table_name} - Columns: {', '.join(columns)}"
                              for table_name, columns in schema.items()])
    result_text = "\n".join([str(record) for record in results])


    # Pull sector from enriched result (assumed last column)
    sector_info = ""
    if results and isinstance(results[0], (list, tuple)) and len(results[0]) >= 3:
        sector_desc = results[0][-1]
        if isinstance(sector_desc, str) and len(sector_desc) > 3:
            sector_info = f"This query is related to the sector: '{sector_desc}'.\n"


    prompt = f"""
    User Query: {query}

    Schema Details:
    {schema_text}

    SQL Results:
    {result_text}

    {sector_info}

    Generate a concise, human-friendly summary of the results in about 50 words. Be specific about the sector or topic.
    """
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    summary = response.choices[0].message.content.strip()
    return summary


def run_pipeline(user_query, db_path):
    """
    Run the full pipeline: dynamically extract the schema, generate SQL queries, execute one that returns results,
    and summarize the findings.
    (Combined approach)
    """
    schemas = fetch_schema_from_db(db_path)
    generated_sql_queries = generate_multiple_sql_queries_with_schema(user_query, schemas)

    for query in generated_sql_queries:
        cleaned_query = clean_sql_query(query)
        sql_query_for_db = revert_aliases_in_query(cleaned_query)
        results = execute_sql_query(db_path, sql_query_for_db)

        if results and not isinstance(results, str):
            try:
                df_ntee = pd.read_csv(ntee_path, encoding='latin1')
                code_to_desc = dict(zip(df_ntee['org_class_code'], df_ntee['class_short_description']))

                # Try to detect a valid NTEE code column
                code_index = None
                for i, val in enumerate(results[0]):
                    if isinstance(val, str) and re.match(r"^[A-Z]\d{2}", val):  # like G41, E20, etc.
                        if val in code_to_desc:
                            code_index = i
                            break

                # Enrich results if possible
                if code_index is not None:
                    enriched_results = []
                    for row in results:
                        row = list(row)
                        code = row[code_index]
                        description = code_to_desc.get(code, "Unknown Sector")
                        row.append(description)
                        enriched_results.append(tuple(row))
                    results = enriched_results
            except Exception as e:
                # Optional: log or handle error
                pass

            # This line should not be indented under the try/except or if
            summary = summarize_findings_with_llm(user_query, schemas, results)
            return cleaned_query, results, summary

    return "No valid results found for any query.", None, None


st.title("CharityBot")
st.markdown("Ask about grants, foundations, or sectors. Get instant answers!")

# Conversation state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_option" not in st.session_state:
    st.session_state.awaiting_option = False
if "top_similar" not in st.session_state:
    st.session_state.top_similar = None
if "user_query" not in st.session_state:
    st.session_state.user_query = None

def get_top_similar_queries(user_query):

    df = pd.read_csv(samplequery_path)
    queries = df['Question'].tolist()
    model_retrieval = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model_retrieval.encode(queries)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    user_embedding = model_retrieval.encode([user_query])
    distances, indices = index.search(user_embedding, 3)
    top_similar = [(queries[idx], 1 / (1 + dist), idx) for dist, idx in zip(distances[0], indices[0])]
    return top_similar, df

# --- Chat Interface ---
def show_message(message, is_user):
    align = "user" if is_user else "assistant"
    st.chat_message(align).markdown(message)

for msg, is_user in st.session_state.messages:
    show_message(msg, is_user)
user_query = None
if not st.session_state.awaiting_option:
    user_query = st.chat_input("Type your question about grants or organizations...", key="main_input")
    if user_query:
        st.session_state.messages.append((user_query, True))
        show_message(user_query, is_user=True)
        # Get similar queries
        top_similar, df = get_top_similar_queries(user_query)
        st.session_state.top_similar = (top_similar, df)
        st.session_state.user_query = user_query
        st.session_state.awaiting_option = True
        st.rerun()
else:
    # Present options for selection
    top_similar, df = st.session_state.top_similar
    options = [q for q, sim, idx in top_similar]
    options.append("None of the above")
    st.markdown("**Did you mean one of these?**")
    selection = st.radio(
        label="Choose a query option:",
        options=options,
        index=len(options)-1,
        key=f"simq_{st.session_state.user_query}",
        label_visibility="collapsed"
    )
    submit = st.button("Submit", key=f"submit_option_{st.session_state.user_query}")
    if submit:
        user_query = st.session_state.user_query
        if selection != "None of the above":
            selected_idx = next(idx for (q, sim, idx) in top_similar if q == selection)
            selected_sql_query = df.iloc[selected_idx]['SQL Query']
            schemas = fetch_schema_from_db(db_path)
            cleaned_sql = clean_sql_query(selected_sql_query)
            results = execute_sql_query(db_path, cleaned_sql)
            summary = summarize_findings_with_llm(user_query, schemas, results)
        else:
            sql_query, results, summary = run_pipeline(user_query, db_path)
        st.session_state.messages.append((summary, False))
        show_message(summary, is_user=False)
        # Reset for next question
        st.session_state.awaiting_option = False
        st.session_state.top_similar = None
        st.session_state.user_query = None
        st.rerun()
