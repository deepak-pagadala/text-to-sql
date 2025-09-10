import { useState, useEffect, useRef } from "react";

// JavaScript version - no interfaces needed
// Data structures:
// SimilarQuery: { question, similarity_score, index }
// ChatResponse: { bot_response, sql_executed?, reformulated_query?, response_type, session_id, timestamp, similar_queries? }
// ChatMessage: { sender: 'user'|'bot', text, timestamp, responseType?, similarQueries?, reformulatedQuery?, sqlExecuted?, originalQuery? }

function ChatInterface() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [sessionReady, setSessionReady] = useState(false);
  
  // Flow state management
  const [pendingUserQuery, setPendingUserQuery] = useState(null);
  const [awaitingSimilarSelection, setAwaitingSimilarSelection] = useState(false);
  const [awaitingBroadConfirmation, setAwaitingBroadConfirmation] = useState(false);
  
  const messagesEndRef = useRef(null);
  const [connectionError, setConnectionError] = useState(false);

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  // Enhanced session management
  useEffect(() => {
    const savedSessionId = localStorage.getItem('charitybot_session_id');
    if (savedSessionId) {
      console.log("Using saved session:", savedSessionId);
      setSessionId(savedSessionId);
      setSessionReady(true);
      testConnection();
      return;
    }
    createNewSession();
  }, []);

  const testConnection = async () => {
    try {
      const response = await fetch("http://localhost:8000/health");
      if (response.ok) {
        setConnectionError(false);
      } else {
        setConnectionError(true);
      }
    } catch (error) {
      setConnectionError(true);
    }
  };

  const createNewSession = async () => {
    try {
      const res = await fetch("http://localhost:8000/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          user_info: "Web user",
          memory_window_size: 4 
        }),
      });
      
      if (res.ok) {
        const data = await res.json();
        setSessionId(data.session_id);
        setSessionReady(true);
        localStorage.setItem('charitybot_session_id', data.session_id);
        setConnectionError(false);
        console.log("New session created:", data.session_id);
      } else {
        console.error("Failed to create session:", res.status);
        setSessionReady(false);
        setConnectionError(true);
      }
    } catch (e) {
      console.error("Session creation failed:", e);
      setSessionReady(false);
      setConnectionError(true);
    }
  };

  const sendMessage = async (messageText, selectedSimilarQuery, confirmBroadQuery) => {
    const queryToSend = messageText || input;
    if (!queryToSend.trim() || !sessionReady || !sessionId) return;
    
    console.log("sendMessage called:", { queryToSend, selectedSimilarQuery, confirmBroadQuery });
    
    // Add user message for new queries (not selections/confirmations)
    if (selectedSimilarQuery === undefined && confirmBroadQuery === undefined) {
      setInput("");
      const userTimestamp = new Date();
      setMessages((msgs) => [...msgs, { 
        sender: "user", 
        text: queryToSend, 
        timestamp: userTimestamp 
      }]);
      setPendingUserQuery(queryToSend);
    }
    
    setLoading(true);

    try {
      const requestBody = {
        user_query: queryToSend,
        session_id: sessionId
      };

      if (selectedSimilarQuery !== undefined) {
        requestBody.selected_similar_query = selectedSimilarQuery;
      }
      if (confirmBroadQuery !== undefined) {
        requestBody.confirm_broad_query = confirmBroadQuery;
      }

      console.log("Sending request:", requestBody);

      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      console.log("Response data:", data);
      
      handleChatResponse(data, queryToSend);
      setConnectionError(false);
      
    } catch (e) {
      console.error("Error in sendMessage:", e);
      const errorTimestamp = new Date();
      setMessages((msgs) => [...msgs, { 
        sender: "bot", 
        text: "Error: Could not reach the server. Please check if the backend is running on port 8000.", 
        timestamp: errorTimestamp,
        responseType: "error"
      }]);
      setConnectionError(true);
    }
    
    setLoading(false);
  };

  const handleChatResponse = (data, originalQuery) => {
    const botTimestamp = new Date();
    
    console.log("Handling response type:", data.response_type, data);
    
    switch (data.response_type) {
      case "similar_questions":
        console.log("Showing similar questions:", data.similar_queries);
        setMessages((msgs) => [...msgs, {
          sender: "bot",
          text: data.bot_response,
          timestamp: botTimestamp,
          responseType: "similar_questions",
          similarQueries: data.similar_queries || [],
          originalQuery: originalQuery
        }]);
        setAwaitingSimilarSelection(true);
        break;
        
      case "broad_query_suggestion":
        console.log("Showing broad query suggestion:", data.reformulated_query);
        setMessages((msgs) => [...msgs, {
          sender: "bot",
          text: data.bot_response,
          timestamp: botTimestamp,
          responseType: "broad_query_suggestion",
          reformulatedQuery: data.reformulated_query,
          originalQuery: originalQuery
        }]);
        setAwaitingBroadConfirmation(true);
        break;
        
      case "cached":
      case "pipeline":
      case "error":
      default:
        setMessages((msgs) => [...msgs, {
          sender: "bot",
          text: data.bot_response,
          timestamp: botTimestamp,
          responseType: data.response_type,
          sqlExecuted: data.sql_executed
        }]);
        
        // Reset flow state
        setAwaitingSimilarSelection(false);
        setAwaitingBroadConfirmation(false);
        setPendingUserQuery(null);
        break;
    }
  };

  const handleSimilarQuerySelection = async (selectedIndex) => {
    if (!sessionReady || !sessionId || !pendingUserQuery) return;
    
    console.log("User selected similar query index:", selectedIndex);
    
    setAwaitingSimilarSelection(false);
    
    if (selectedIndex === -1) {
      // User selected "none of the above"
      const userTimestamp = new Date();
      setMessages((msgs) => [...msgs, {
        sender: "user",
        text: "None of the above - proceed with my original question",
        timestamp: userTimestamp
      }]);
    } else {
      // User selected a similar query
      const lastBotMessage = messages[messages.length - 1];
      if (lastBotMessage?.similarQueries && lastBotMessage.similarQueries[selectedIndex]) {
        const selectedQuery = lastBotMessage.similarQueries[selectedIndex];
        const userTimestamp = new Date();
        setMessages((msgs) => [...msgs, {
          sender: "user",
          text: selectedQuery.question,
          timestamp: userTimestamp
        }]);
      }
    }
    
    await sendMessage(pendingUserQuery, selectedIndex);
  };

  const handleBroadQueryConfirmation = async (confirmed) => {
    if (!sessionReady || !sessionId || !pendingUserQuery) return;
    
    console.log("User confirmed broad query:", confirmed);
    
    setAwaitingBroadConfirmation(false);
    
    const userTimestamp = new Date();
    const responseText = confirmed 
      ? "Yes, proceed with the suggestion" 
      : "No, let me rephrase my question";
    
    setMessages((msgs) => [...msgs, {
      sender: "user",
      text: responseText,
      timestamp: userTimestamp
    }]);
    
    await sendMessage(pendingUserQuery, undefined, confirmed);
    
    if (!confirmed) {
      setPendingUserQuery(null);
    }
  };

  const clearChat = async () => {
    setMessages([]);
    setInput("");
    setPendingUserQuery(null);
    setAwaitingSimilarSelection(false);
    setAwaitingBroadConfirmation(false);
    
    localStorage.removeItem('charitybot_session_id');
    setSessionId(null);
    setSessionReady(false);
    
    await createNewSession();
  };

  const formatTime = (date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const formatResponseText = (text) => {
    if (!text) return "";
    
    let formattedText = text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`(.*?)`/g, '<code>$1</code>');
    
    const paragraphs = formattedText.split('\n\n').filter(p => p.trim());
    
    return paragraphs.map((paragraph, idx) => (
      <p key={idx} style={{ 
        margin: idx === 0 ? '0 0 12px 0' : '12px 0', 
        lineHeight: '1.6',
        fontSize: '1rem'
      }}>
        <span dangerouslySetInnerHTML={{ __html: paragraph }} />
      </p>
    ));
  };

  const renderMessage = (msg, index) => {
    // Similar questions selection
    if (msg.responseType === "similar_questions" && msg.similarQueries) {
      return (
        <div
          key={`${msg.sender}-${index}-similar`}
          style={{
            position: "relative",
            backgroundColor: "rgba(33, 136, 56, 0.1)",
            padding: "20px",
            borderRadius: "12px",
            marginBottom: "20px",
            border: "1px solid rgba(33, 136, 56, 0.2)",
            boxShadow: "0 2px 8px rgba(15, 81, 50, 0.1)"
          }}
        >
          <div style={{ 
            position: "absolute", 
            top: "-8px", 
            left: "12px",
            fontSize: "0.75rem", 
            color: "#17412d", 
            fontWeight: 600,
            background: "rgba(33, 136, 56, 0.1)",
            padding: "2px 8px",
            borderRadius: "8px",
            boxShadow: "0 1px 3px rgba(15, 81, 50, 0.15)",
            border: "1px solid rgba(33, 136, 56, 0.2)"
          }}>
            CharityBot
          </div>

          <div style={{ marginTop: "8px", marginBottom: "16px", color: "#2d5a3d" }}>
            {msg.text}
          </div>
          
          <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            {msg.similarQueries.map((query, qIndex) => (
              <div
                key={qIndex}
                style={{
                  background: "#e8f5e8",
                  padding: "12px 16px",
                  borderRadius: "8px",
                  border: "1px solid #c3e6c3",
                  cursor: sessionReady ? "pointer" : "not-allowed",
                  opacity: sessionReady ? 1 : 0.6,
                  transition: "all 0.2s ease",
                  display: "flex",
                  alignItems: "center",
                  gap: "12px"
                }}
                onClick={() => sessionReady && handleSimilarQuerySelection(qIndex)}
                onMouseEnter={(e) => {
                  if (sessionReady) {
                    e.currentTarget.style.transform = "translateY(-1px)";
                    e.currentTarget.style.boxShadow = "0 2px 8px rgba(0,0,0,0.1)";
                  }
                }}
                onMouseLeave={(e) => {
                  if (sessionReady) {
                    e.currentTarget.style.transform = "translateY(0)";
                    e.currentTarget.style.boxShadow = "none";
                  }
                }}
              >
                <div style={{
                  background: "#218838",
                  color: "white",
                  borderRadius: "50%",
                  width: "24px",
                  height: "24px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "0.8rem",
                  fontWeight: "600",
                  flexShrink: 0
                }}>
                  {qIndex + 1}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ 
                    fontSize: "0.9rem", 
                    color: "#155724",
                    lineHeight: "1.4",
                    marginBottom: "4px"
                  }}>
                    {query.question}
                  </div>
                  <div style={{ 
                    fontSize: "0.75rem", 
                    color: "#6c757d" 
                  }}>
                    Similarity: {(query.similarity_score * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            ))}
            
            <div
              style={{
                background: "#f8f9fa",
                padding: "12px 16px",
                borderRadius: "8px",
                border: "1px solid #e9ecef",
                cursor: sessionReady ? "pointer" : "not-allowed",
                opacity: sessionReady ? 1 : 0.6,
                transition: "all 0.2s ease",
                display: "flex",
                alignItems: "center",
                gap: "12px"
              }}
              onClick={() => sessionReady && handleSimilarQuerySelection(-1)}
              onMouseEnter={(e) => {
                if (sessionReady) {
                  e.currentTarget.style.transform = "translateY(-1px)";
                  e.currentTarget.style.boxShadow = "0 2px 8px rgba(0,0,0,0.1)";
                }
              }}
              onMouseLeave={(e) => {
                if (sessionReady) {
                  e.currentTarget.style.transform = "translateY(0)";
                  e.currentTarget.style.boxShadow = "none";
                }
              }}
            >
              <div style={{
                background: "#6c757d",
                color: "white",
                borderRadius: "50%",
                width: "24px",
                height: "24px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "0.8rem",
                fontWeight: "600",
                flexShrink: 0
              }}>
                {msg.similarQueries.length + 1}
              </div>
              <div style={{
                fontSize: "0.9rem",
                color: "#495057",
                fontWeight: "500"
              }}>
                None of the above
              </div>
            </div>
          </div>
          
          <div style={{ 
            fontSize: "0.7rem", 
            color: "rgba(0, 0, 0, 0.4)", 
            marginTop: "8px", 
            textAlign: "right",
            fontStyle: "italic"
          }}>
            {formatTime(msg.timestamp)}
          </div>
        </div>
      );
    }

    // Broad query confirmation
    if (msg.responseType === "broad_query_suggestion") {
      return (
        <div
          key={`${msg.sender}-${index}-broad`}
          style={{
            position: "relative",
            backgroundColor: "rgba(33, 136, 56, 0.1)",
            padding: "20px",
            borderRadius: "12px",
            marginBottom: "20px",
            border: "1px solid rgba(33, 136, 56, 0.2)",
            boxShadow: "0 2px 8px rgba(15, 81, 50, 0.1)"
          }}
        >
          <div style={{ 
            position: "absolute", 
            top: "-8px", 
            left: "12px",
            fontSize: "0.75rem", 
            color: "#17412d", 
            fontWeight: 600,
            background: "rgba(33, 136, 56, 0.1)",
            padding: "2px 8px",
            borderRadius: "8px",
            boxShadow: "0 1px 3px rgba(15, 81, 50, 0.15)",
            border: "1px solid rgba(33, 136, 56, 0.2)"
          }}>
            CharityBot
          </div>
          
          <div style={{ marginTop: "8px", marginBottom: "16px", color: "#2d5a3d" }}>
            {formatResponseText(msg.text)}
          </div>
          
          <div style={{
            display: "flex",
            gap: "12px",
            flexWrap: "wrap"
          }}>
            <button
              onClick={() => sessionReady && handleBroadQueryConfirmation(true)}
              disabled={!sessionReady}
              style={{
                background: sessionReady ? "#28a745" : "#6c757d",
                color: "white",
                border: "none",
                borderRadius: "8px",
                padding: "10px 20px",
                fontSize: "0.9rem",
                fontWeight: 600,
                cursor: sessionReady ? "pointer" : "not-allowed",
                opacity: sessionReady ? 1 : 0.6,
                transition: "all 0.2s ease",
                flex: "1",
                minWidth: "140px"
              }}
            >
              {sessionReady ? "Yes, proceed" : "Initializing..."}
            </button>
            <button
              onClick={() => sessionReady && handleBroadQueryConfirmation(false)}
              disabled={!sessionReady}
              style={{
                background: "#6c757d",
                color: "white",
                border: "none",
                borderRadius: "8px",
                padding: "10px 20px",
                fontSize: "0.9rem",
                fontWeight: 600,
                cursor: sessionReady ? "pointer" : "not-allowed",
                opacity: sessionReady ? 1 : 0.6,
                transition: "all 0.2s ease",
                flex: "1",
                minWidth: "140px"
              }}
            >
              {sessionReady ? "No, let me rephrase" : "Initializing..."}
            </button>
          </div>
          
          <div style={{ 
            fontSize: "0.7rem", 
            color: "rgba(0, 0, 0, 0.4)", 
            marginTop: "8px", 
            textAlign: "right",
            fontStyle: "italic"
          }}>
            {formatTime(msg.timestamp)}
          </div>
        </div>
      );
    }

    // Regular message - styled like original premier bubbles
    const isUser = msg.sender === "user";
    const bubbleStyle = isUser ? {
      background: "linear-gradient(135deg, #218838 0%, #28a745 100%)",
      color: "white",
      marginLeft: "auto",
      marginRight: "0",
      borderRadius: "18px 18px 4px 18px"
    } : {
      background: "rgba(33, 136, 56, 0.1)",
      color: "#2d5a3d",
      marginLeft: "0",
      marginRight: "auto",
      borderRadius: "18px 18px 18px 4px",
      border: "1px solid rgba(33, 136, 56, 0.2)"
    };
    
    return (
      <div
        key={`${msg.sender}-${index}-regular`}
        style={{
          ...bubbleStyle,
          position: "relative",
          maxWidth: "75%",
          padding: "16px 20px",
          marginBottom: "16px",
          boxShadow: "0 2px 8px rgba(15, 81, 50, 0.1)"
        }}
      >
        <div style={{ 
          position: "absolute", 
          top: "-8px", 
          left: isUser ? "auto" : "12px",
          right: isUser ? "12px" : "auto",
          fontSize: "0.75rem", 
          color: "#17412d", 
          fontWeight: 600,
          background: "rgba(33, 136, 56, 0.1)",
          padding: "2px 8px",
          borderRadius: "8px",
          boxShadow: "0 1px 3px rgba(15, 81, 50, 0.15)",
          border: "1px solid rgba(33, 136, 56, 0.2)"
        }}>
          {isUser ? "You" : "CharityBot"}
          {msg.responseType === "cached" && (
            <span style={{ marginLeft: "4px", fontSize: "0.7rem" }}>💾</span>
          )}
        </div>
        
        <div style={{ marginTop: "8px" }}>
          {isUser ? msg.text : formatResponseText(msg.text)}
        </div>
        
        {msg.sqlExecuted && (
          <details style={{ marginTop: "8px", fontSize: "0.8rem", color: isUser ? "rgba(255,255,255,0.8)" : "#6c757d" }}>
            <summary style={{ cursor: "pointer", fontWeight: "500" }}>View SQL Query</summary>
            <pre style={{ 
              background: "#f8f9fa", 
              padding: "8px", 
              borderRadius: "4px", 
              marginTop: "4px",
              overflow: "auto",
              fontSize: "0.75rem",
              color: "#333"
            }}>
              {msg.sqlExecuted}
            </pre>
          </details>
        )}
        
        <div style={{ 
          fontSize: "0.7rem", 
          color: isUser ? "rgba(255,255,255,0.7)" : "rgba(0, 0, 0, 0.4)", 
          marginTop: "8px", 
          textAlign: "right",
          fontStyle: "italic"
        }}>
          {formatTime(msg.timestamp)}
        </div>
      </div>
    );
  };

  return (
    <div style={{ display: "flex", minHeight: "100vh", background: "#ffffff" }}>
      {/* Left Description Section */}
      <div style={{ 
        flex: 1, 
        display: "flex", 
        flexDirection: "column", 
        justifyContent: "center", 
        alignItems: "flex-start", 
        padding: "0 3rem 0 4rem", 
        maxWidth: 550 
      }}>
        <h1 style={{ 
          color: '#17412d', 
          fontSize: '3rem', 
          fontWeight: 800, 
          marginBottom: 24, 
          letterSpacing: 1.5,
          lineHeight: 1.1 
        }}>
          Smart Charity
        </h1>
        
        <div style={{ 
          color: '#218838', 
          fontSize: '1.3rem', 
          fontWeight: 600, 
          marginBottom: 32,
          lineHeight: 1.4 
        }}>
          Your Philanthropic Grant Data AI Assistant
        </div>
        
        <div style={{ 
          color: '#2d5a3d', 
          fontSize: '1.15rem', 
          lineHeight: 1.6, 
          marginBottom: 28 
        }}>
          <strong style={{ color: '#17412d' }}>Smart Charity</strong> helps you explore and analyze nonprofit financial and organizational data with ease.
        </div>
        
        <ul style={{ 
          listStyle: 'none', 
          padding: 0, 
          margin: 0,
          fontSize: '1.1rem',
          lineHeight: 2.2 
        }}>
          <li style={{ color: '#218838', fontWeight: 500, marginBottom: 8 }}>
            • Ask about organizations, grants, and trends
          </li>
          <li style={{ color: '#218838', fontWeight: 500, marginBottom: 8 }}>
            • Get clear, accurate, and insightful answers
          </li>
          <li style={{ color: '#218838', fontWeight: 500 }}>
            • Make informed decisions with advanced AI support
          </li>
        </ul>
        
        {connectionError && (
          <div style={{
            marginTop: "32px",
            padding: "16px",
            background: "#fef2f2",
            border: "1px solid #fecaca",
            borderRadius: "8px",
            color: "#dc2626"
          }}>
            <p style={{ fontWeight: "600", margin: "0 0 8px 0" }}>⚠️ Backend Connection Error</p>
            <p style={{ fontSize: "0.9rem", margin: 0 }}>
              Make sure the backend server is running on port 8000
            </p>
          </div>
        )}
      </div>

      {/* Right Chat Section */}
      <div style={{ 
        flex: 1.2, 
        display: "flex", 
        alignItems: "center", 
        justifyContent: "flex-end", 
        background: "transparent", 
        width: "100%", 
        paddingRight: "2rem" 
      }}>
        <div style={{ 
          marginRight: 0, 
          marginLeft: "-50px", 
          width: "100%", 
          maxWidth: "1350px",
          display: "flex",
          flexDirection: "column",
          height: "90vh",
          background: "rgba(248, 250, 252, 0.8)",
          borderRadius: "16px",
          border: "1px solid rgba(33, 136, 56, 0.1)",
          boxShadow: "0 8px 32px rgba(15, 81, 50, 0.1)"
        }}>
          <div style={{
            flex: 1,
            overflowY: "auto",
            padding: "20px",
            display: "flex",
            flexDirection: "column"
          }}>
            {/* Session Status Indicator */}
            {sessionId && (
              <div style={{
                textAlign: "center",
                padding: "8px",
                margin: "8px 0",
                fontSize: "0.75rem",
                color: "#6c757d",
                background: "rgba(33, 136, 56, 0.05)",
                borderRadius: "4px",
                border: "1px solid rgba(33, 136, 56, 0.1)"
              }}>
                💬 Session active - CharityBot will remember our conversation
              </div>
            )}
            
            {messages.map((msg, i) => renderMessage(msg, i))}
            
            {loading && (
              <div style={{
                background: "rgba(33, 136, 56, 0.1)",
                padding: "16px 20px",
                borderRadius: "18px 18px 18px 4px",
                marginBottom: "16px",
                maxWidth: "75%",
                border: "1px solid rgba(33, 136, 56, 0.2)"
              }}>
                <i style={{ color: "#2d5a3d" }}>CharityBot is typing...</i>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          <div style={{ 
            width: "100%", 
            maxWidth: "1250px", 
            padding: "20px",
            borderTop: "1px solid rgba(33, 136, 56, 0.1)",
            background: "rgba(255, 255, 255, 0.9)"
          }}>
            <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { 
                  if (e.key === "Enter" && !awaitingSimilarSelection && !awaitingBroadConfirmation) {
                    sendMessage();
                  }
                }}
                placeholder={
                  awaitingSimilarSelection ? "Please select from the options above..." :
                  awaitingBroadConfirmation ? "Please choose to proceed or rephrase..." :
                  sessionReady ? "Type your message..." : "Initializing session..."
                }
                disabled={loading || !sessionReady || awaitingSimilarSelection || awaitingBroadConfirmation}
                suppressHydrationWarning
                style={{ 
                  flex: "1 1 0%", 
                  minWidth: "0", 
                  marginRight: "12px",
                  padding: "12px 16px",
                  border: "2px solid rgba(33, 136, 56, 0.2)",
                  borderRadius: "12px",
                  fontSize: "1rem",
                  outline: "none",
                  transition: "border-color 0.2s ease",
                  background: "white"
                }}
                onFocus={(e) => e.target.style.borderColor = "#218838"}
                onBlur={(e) => e.target.style.borderColor = "rgba(33, 136, 56, 0.2)"}
              />
              <button 
                onClick={() => sendMessage()} 
                disabled={loading || !input.trim() || !sessionReady || awaitingSimilarSelection || awaitingBroadConfirmation}
                suppressHydrationWarning
                style={{
                  background: sessionReady && input.trim() ? "linear-gradient(135deg, #218838 0%, #28a745 100%)" : "#6c757d",
                  color: "white",
                  border: "none",
                  borderRadius: "12px",
                  padding: "12px 24px",
                  fontSize: "1rem",
                  fontWeight: 600,
                  cursor: sessionReady && input.trim() ? "pointer" : "not-allowed",
                  transition: "all 0.2s ease",
                  boxShadow: sessionReady && input.trim() ? "0 2px 8px rgba(33, 136, 56, 0.3)" : "none"
                }}
              >
                {sessionReady ? "Submit" : "Initializing..."}
              </button>
              <button
                onClick={clearChat}
                disabled={messages.length === 0}
                style={{
                  background: messages.length > 0 ? "#dc3545" : "#6c757d",
                  color: "white",
                  border: "none",
                  borderRadius: "12px",
                  padding: "12px 16px",
                  fontSize: "1rem",
                  cursor: messages.length > 0 ? "pointer" : "not-allowed",
                  transition: "all 0.2s ease"
                }}
                title="Clear conversation"
              >
                🗑️ Clear
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ChatPage() {
  return <ChatInterface />;
}