from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime

# Import your existing chatbot functions
from mistral_chatbotV2 import (
    create_new_session,
    get_session_context,
    add_chat_to_session,
    get_top_similar_queries,
    quick_reframe,
    enhanced_pipeline_with_smart_followup_detection,  # Add this
    execute_sql_query,
    summarize_findings_with_llm,
    fetch_schema_from_db,
    TOP_K_DEFAULT,
    DB_PATH
)


app = FastAPI(title="CharityBot API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class CreateSessionRequest(BaseModel):
    user_info: Optional[str] = "Web user"
    memory_window_size: Optional[int] = 4

class CreateSessionResponse(BaseModel):
    session_id: str
    message: str

class ChatRequest(BaseModel):
    user_query: str
    session_id: str
    selected_similar_query: Optional[int] = None  # -1 for "none of the above"
    confirm_broad_query: Optional[bool] = None

class SimilarQuery(BaseModel):
    question: str
    similarity_score: float
    index: int

class ChatResponse(BaseModel):
    bot_response: str
    sql_executed: Optional[str] = None
    reformulated_query: Optional[str] = None
    response_type: str  # "similar_questions", "broad_query_suggestion", "cached", "pipeline", "error"
    session_id: str
    timestamp: str
    similar_queries: Optional[List[SimilarQuery]] = None
    original_query: Optional[str] = None  # Add this to show what was reformulated

class SessionContextResponse(BaseModel):
    session_id: str
    recent_chats: List[Dict[str, Any]]
    chat_summaries: List[Dict[str, Any]]
    total_chats: int

# API Endpoints

@app.post("/sessions", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new chat session."""
    try:
        session_id = create_new_session(request.user_info, request.memory_window_size)
        if not session_id:
            raise HTTPException(status_code=500, detail="Failed to create session")
        
        return CreateSessionResponse(
            session_id=session_id,
            message="Session created successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

@app.get("/sessions/{session_id}/context", response_model=SessionContextResponse)
async def get_session_context_api(session_id: str):
    """Get conversation context for a session."""
    try:
        context = get_session_context(session_id)
        return SessionContextResponse(
            session_id=context['session_id'],
            recent_chats=[{
                'chat_index': chat[0],
                'user_query': chat[1],
                'expanded_query': chat[2],
                'bot_response': chat[3],
                'sql_executed': chat[4],
                'timestamp': chat[5]
            } for chat in context['recent_chats']],
            chat_summaries=[{
                'summary_text': summary[0],
                'covers_chat_start': summary[1],
                'covers_chat_end': summary[2]
            } for summary in context['chat_summaries']],
            total_chats=context['total_chats']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving context: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint that handles all conversation scenarios."""
    try:
        timestamp = datetime.now().isoformat()
        
        # Handle similar query selection
        if request.selected_similar_query is not None:
            return await handle_similar_query_selection(request, timestamp)
        
        # Handle broad query confirmation
        if request.confirm_broad_query is not None:
            return await handle_broad_query_confirmation(request, timestamp)
        
        # Handle new user query
        return await handle_new_query(request, timestamp)
        
    except Exception as e:
        error_response = ChatResponse(
            bot_response=f"I encountered an error processing your request: {str(e)}. Please try again.",
            response_type="error",
            session_id=request.session_id,
            timestamp=timestamp
        )
        
        # Log error to session
        try:
            add_chat_to_session(
                request.session_id, 
                request.user_query, 
                request.user_query,
                error_response.bot_response, 
                None, 
                'error'
            )
        except:
            pass  # Don't fail on logging errors
            
        return error_response

async def handle_similar_query_selection(request: ChatRequest, timestamp: str) -> ChatResponse:
    """Handle when user selects from similar queries."""
    if request.selected_similar_query == -1:
        # User chose "none of the above" - proceed with original query using web-compatible pipeline
        sql_executed, results, summary, reformulated_query =  enhanced_pipeline_with_smart_followup_detection(
            request.user_query, request.session_id, DB_PATH
        )
        
        # If query was reformulated, show this to the user
        if reformulated_query and reformulated_query != request.user_query:
            summary = f"I understood your question as: \"{reformulated_query}\"\n\n{summary}"
    else:
        # User selected a specific similar query - execute it
        tops, df = get_top_similar_queries(request.user_query)
        if request.selected_similar_query < len(tops):
            selected_question = tops[request.selected_similar_query][0]
            selected_idx = tops[request.selected_similar_query][2]
            
            # Get cached SQL and PROCESS IT PROPERLY
            canned_sql = df.iloc[selected_idx]["SQL Query"]
            
            # ADD THESE MISSING STEPS:
            from mistral_chatbotV2 import clean_sql_query, revert_aliases_in_query, replace_table_aliases
            
            cleaned = clean_sql_query(canned_sql)
            processed_sql = revert_aliases_in_query(cleaned)
            processed_sql = replace_table_aliases(processed_sql)
            
            schemas = fetch_schema_from_db(DB_PATH)
            results = execute_sql_query(processed_sql, DB_PATH)  # Use processed_sql, not canned_sql
            summary = summarize_findings_with_llm(selected_question, schemas, results)
            sql_executed = processed_sql  # Return processed version
            reformulated_query = None
        else:
            raise HTTPException(status_code=400, detail="Invalid similar query selection")
    
    # Save to session
    response_type = "cached" if request.selected_similar_query != -1 else "pipeline"
    add_chat_to_session(
        request.session_id, 
        request.user_query, 
        reformulated_query or request.user_query,
        summary, 
        sql_executed, 
        response_type
    )
    
    return ChatResponse(
        bot_response=summary,
        sql_executed=sql_executed,
        response_type=response_type,
        session_id=request.session_id,
        timestamp=timestamp,
        reformulated_query=reformulated_query,
        original_query=request.user_query if reformulated_query else None
    )
async def handle_broad_query_confirmation(request: ChatRequest, timestamp: str) -> ChatResponse:
    """Handle broad query confirmation."""
    if request.confirm_broad_query:
        # User confirmed - proceed with reframed query using web-compatible pipeline
        reframed_query = quick_reframe(request.user_query, TOP_K_DEFAULT)
        sql_executed, results, summary, reformulated_query =  enhanced_pipeline_with_smart_followup_detection(
            reframed_query, request.session_id, DB_PATH
        )
        
        # Show that we used the reframed version
        summary = f"Using reframed query: \"{reframed_query}\"\n\n{summary}"
    else:
        # User declined - ask them to rephrase
        summary = "Please rephrase your question to be more specific. For example, you could ask for 'top 10' results or focus on a particular time period, location, or sector."
        sql_executed = None
        reformulated_query = None
    
    add_chat_to_session(
        request.session_id, 
        request.user_query, 
        reformulated_query or request.user_query,
        summary, 
        sql_executed, 
        'pipeline'
    )
    
    return ChatResponse(
        bot_response=summary,
        sql_executed=sql_executed,
        response_type="pipeline",
        session_id=request.session_id,
        timestamp=timestamp,
        reformulated_query=reformulated_query,
        original_query=request.user_query if reformulated_query else None
    )
async def handle_new_query(request: ChatRequest, timestamp: str) -> ChatResponse:
    """Handle a brand new user query."""
    # First check for similar queries
    tops, df = get_top_similar_queries(request.user_query, k=3)
    
    if tops and tops[0][1] > 0.5:
        similar_queries = [
            SimilarQuery(
                question=q,
                similarity_score=score,
                index=idx
            ) for q, score, idx in tops
        ]
        
        return ChatResponse(
            bot_response="I found some similar questions that might help. Please select one, or choose 'None of the above' to proceed with your original question:",
            response_type="similar_questions",
            session_id=request.session_id,
            timestamp=timestamp,
            similar_queries=similar_queries
        )
    
    
    # Use the web-compatible enhanced pipeline with context-aware reframing
    try:
        sql_executed, results, summary, reformulated_query =  enhanced_pipeline_with_smart_followup_detection(
            request.user_query, request.session_id, DB_PATH
        )
        
        if not summary:
            summary = "I couldn't find relevant results for that query. Please try rephrasing or being more specific."
            response_type = "error"
        else:
            response_type = "pipeline"
        
        # If query was reformulated, show this to the user
        if reformulated_query and reformulated_query != request.user_query:
            summary = f"I understood your question as: \"{reformulated_query}\"\n\n{summary}"
            
        add_chat_to_session(
            request.session_id, 
            request.user_query, 
            reformulated_query or request.user_query,  # Store the working query
            summary, 
            sql_executed, 
            response_type
        )
        
        return ChatResponse(
            bot_response=summary,
            sql_executed=sql_executed,
            response_type=response_type,
            session_id=request.session_id,
            timestamp=timestamp,
            reformulated_query=reformulated_query,
            original_query=request.user_query if reformulated_query else None
        )
        
    except Exception as e:
        error_msg = f"I encountered an error processing your query: {str(e)}"
        add_chat_to_session(
            request.session_id, 
            request.user_query, 
            request.user_query,
            error_msg, 
            None, 
            'error'
        )
        
        return ChatResponse(
            bot_response=error_msg,
            response_type="error",
            session_id=request.session_id,
            timestamp=timestamp
        )
    
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")