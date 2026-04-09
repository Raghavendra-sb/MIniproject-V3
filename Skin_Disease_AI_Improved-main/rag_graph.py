import os
from typing import Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI  # Switched to Gemini

load_dotenv()

# ─────────────────────────────────────────────
# Gemini LLM Setup
# ─────────────────────────────────────────────
# Change from "gemini-1.5-flash" to "gemini-2.5-flash"
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    timeout=60, # Give it 60 seconds to respond before failing
    max_retries=3, # Let the internal SDK try again before LangGraph does
    temperature=0.7
)
# ─────────────────────────────────────────────
# Qdrant + Embeddings
# ─────────────────────────────────────────────
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="skin_disease_vectors",
    embedding=embedding_model,
)

# ─────────────────────────────────────────────
# State definition
# ─────────────────────────────────────────────
class SkinDiseaseState(TypedDict):
    disease_label: str
    confidence_score: float
    retrieved_context: str | None
    llm_explanation: str | None
    is_sufficient: bool | None
    retry_count: int
    final_response: dict | None

# ─────────────────────────────────────────────
# Node 1: Check CNN confidence score
# ─────────────────────────────────────────────
def check_confidence(state: SkinDiseaseState) -> SkinDiseaseState:
    print(f"⚙️  Checking confidence for: {state['disease_label']} ({state['confidence_score']:.2%})")
    return state

def route_confidence(state: SkinDiseaseState) -> Literal["retrieve_context", "low_confidence_response"]:
    if state["confidence_score"] < 0.55:
        print("⚠️  Low confidence — routing to fallback response")
        return "low_confidence_response"
    print("✅  Confidence OK — routing to RAG retrieval")
    return "retrieve_context"

# ─────────────────────────────────────────────
# Node 2a: Low confidence fallback
# ─────────────────────────────────────────────
def low_confidence_response(state: SkinDiseaseState) -> SkinDiseaseState:
    print("💬 Generating low-confidence fallback response...")
    state["final_response"] = {
        "disease_label": state["disease_label"],
        "confidence_score": state["confidence_score"],
        "llm_explanation": (
            f"The model detected a possible case of {state['disease_label']} "
            f"but with low confidence ({state['confidence_score']:.1%}). "
            "Please upload a clearer, well-lit image of the affected skin area "
            "for a more accurate analysis. Consult a dermatologist for professional diagnosis."
        ),
        "retrieved_context": None,
        "warning": "low_confidence",
    }
    return state

# ─────────────────────────────────────────────
# Node 2b: RAG retrieval from Qdrant
# ─────────────────────────────────────────────
def retrieve_context(state: SkinDiseaseState) -> SkinDiseaseState:
    disease = state["disease_label"]
    retry = state.get("retry_count", 0)

    if retry > 0:
        query = f"skin disease symptoms treatment prevention general dermatology {disease}"
        print(f"🔄 Retry {retry}: broader query for {disease}")
    else:
        query = f"{disease} skin disease causes symptoms treatments prevention"
        print(f"🔍 RAG retrieval for: {disease}")

    search_results = vector_db.similarity_search(query=query, k=4)

    context = "\n\n".join([
        f"Page Content: {result.page_content}\nPage Number: {result.metadata.get('page_label', 'N/A')}"
        for result in search_results
    ])

    state["retrieved_context"] = context
    print(f"📄 Retrieved {len(search_results)} chunks from Qdrant")
    return state

# ─────────────────────────────────────────────
# Node 3: LLM explanation via Gemini
# ─────────────────────────────────────────────
def generate_explanation(state: SkinDiseaseState) -> SkinDiseaseState:
    print("🤖 Generating explanation via Gemini...")

    disease    = state["disease_label"]
    context    = state["retrieved_context"]
    confidence = state["confidence_score"]

    # Gemini handles structured lists of messages natively
    messages = [
        (
            "system", 
            "You are a medical AI assistant specializing in dermatology. "
            "Use ONLY the provided context to answer. "
            "Always recommend consulting a qualified dermatologist."
        ),
        (
            "user", 
            f"CONTEXT:\n{context}\n\n"
            f"The CNN model detected: {disease} (Confidence: {confidence:.1%})\n\n"
            f"Please provide:\n"
            f"1. What {disease} is\n"
            f"2. Common causes\n"
            f"3. Key symptoms\n"
            f"4. Recommended treatments\n"
            f"5. Precautions and prevention\n"
            f"6. When to seek medical attention\n\n"
            f"Be empathetic and easy to understand."
        )
    ]

    try:
        # LangChain's invoke method returns an AIMessage object
        response = llm.invoke(messages)
        state["llm_explanation"] = response.content.strip()
        print("✅ Gemini explanation generated.")

    except Exception as e:
        print(f"⚠️ Gemini error: {e}")
        state["llm_explanation"] = (
            f"AI explanation temporarily unavailable.\n\n"
            f"Detected: {disease} — Confidence: {confidence:.1%}\n\n"
            f"Please consult a qualified dermatologist for proper diagnosis."
        )

    return state

# ─────────────────────────────────────────────
# Node 4: Validate LLM response quality
# ─────────────────────────────────────────────
def validate_response(state: SkinDiseaseState) -> SkinDiseaseState:
    print("🔎 Validating response quality...")
    explanation = state["llm_explanation"] or ""
    word_count = len(explanation.split())
    if word_count > 80:
        state["is_sufficient"] = True
        print(f"📊 Sufficient — {word_count} words.")
    else:
        state["is_sufficient"] = False
        print(f"📊 Too short ({word_count} words).")
    return state

# ─────────────────────────────────────────────
# Router: after validate_response
# ─────────────────────────────────────────────
def route_validation(state: SkinDiseaseState) -> Literal["build_final_response", "increment_retry"]:
    retry = state.get("retry_count", 0)
    if not state["is_sufficient"] and retry < 2:
        print(f"🔄 Response insufficient (retry_count={retry}), will retry...")
        return "increment_retry"
    print("✅ Moving to final response.")
    return "build_final_response"

# ─────────────────────────────────────────────
# Node 4b: Increment retry counter
# ─────────────────────────────────────────────
def increment_retry(state: SkinDiseaseState) -> SkinDiseaseState:
    state["retry_count"] = state.get("retry_count", 0) + 1
    print(f"🔁 Retry attempt {state['retry_count']}")
    return state

# ─────────────────────────────────────────────
# Node 5: Build final structured response
# ─────────────────────────────────────────────
def build_final_response(state: SkinDiseaseState) -> SkinDiseaseState:
    print("📦 Building final response...")
    state["final_response"] = {
        "disease_label": state["disease_label"],
        "confidence_score": state["confidence_score"],
        "llm_explanation": state["llm_explanation"],
        "retrieved_context": state["retrieved_context"],
        "warning": None,
    }
    return state

# ─────────────────────────────────────────────
# Build the LangGraph
# ─────────────────────────────────────────────
graph_builder = StateGraph(SkinDiseaseState)

graph_builder.add_node("check_confidence",       check_confidence)
graph_builder.add_node("low_confidence_response", low_confidence_response)
graph_builder.add_node("retrieve_context",         retrieve_context)
graph_builder.add_node("generate_explanation",    generate_explanation)
graph_builder.add_node("validate_response",       validate_response)
graph_builder.add_node("increment_retry",         increment_retry)
graph_builder.add_node("build_final_response",    build_final_response)

graph_builder.add_edge(START,                      "check_confidence")
graph_builder.add_conditional_edges("check_confidence",  route_confidence)
graph_builder.add_edge("low_confidence_response",  END)
graph_builder.add_edge("retrieve_context",         "generate_explanation")
graph_builder.add_edge("generate_explanation",     "validate_response")
graph_builder.add_conditional_edges("validate_response", route_validation)
graph_builder.add_edge("increment_retry",          "retrieve_context")
graph_builder.add_edge("build_final_response",     END)

graph = graph_builder.compile()

# ─────────────────────────────────────────────
# Public function called by Flask app.py
# ─────────────────────────────────────────────
def run_skin_disease_graph(disease_label: str, confidence_score: float) -> dict:
    print(f"\n===== LangGraph Pipeline Start =====")
    print(f"Disease: {disease_label} | Confidence: {confidence_score:.2%}")

    initial_state: SkinDiseaseState = {
        "disease_label":     disease_label,
        "confidence_score": confidence_score,
        "retrieved_context": None,
        "llm_explanation":  None,
        "is_sufficient":    None,
        "retry_count":      0,
        "final_response":   None,
    }

    result = graph.invoke(initial_state)
    print("===== LangGraph Pipeline End =====\n")
    return result["final_response"]