"""
Architecture:
1. ChatGraph (sync): Handles reasoning and response generation
2. IngestionGraph (async): Handles file processing and embedding
3. Clear separation: ChatGraph READS, IngestionGraph WRITES
"""

from typing import TypedDict, Optional, Literal, Any
from langchain_core.tools import BaseTool
from typing_extensions import Annotated
from langchain_core.messages import BaseMessage
import operator
from datetime import datetime

# ============================================================
# FILE PROCESSING TYPES
# ============================================================
class FileReference(TypedDict):
    """Lightweight reference to a file (used in ChatGraph)"""
    file_id: str
    s3_uri: str
    mime_type: str
    status: Literal["pending", "processing", "ready", "failed"]
    
    # Quick access without hitting DB
    thumbnail_uri: Optional[str]
    processing_progress: float  # 0.0 to 1.0

class ProcessedContent(TypedDict):
    """What the LLM actually receives about a file"""
    file_id: str
    content_type: Literal["text", "image_description", "video_summary", "audio_transcript"]
    
    # The actual content (already extracted)
    text_content: Optional[str]
    image_uri: Optional[str]  # For vision model
    
    # Metadata for context
    source_file: str
    timestamp: Optional[str]
    confidence: Optional[float]

class RetrievedChunk(TypedDict):
    """Vector DB retrieval result"""
    chunk_id: str
    content: str
    source_file_id: str
    relevance_score: float
    
    # Context for LLM
    chunk_type: Literal["text", "image_caption", "video_frame", "audio_segment"]
    metadata: dict[str, Any]

# ============================================================
# MODEL SELECTION TYPES
# ============================================================
class ModelCapabilities(TypedDict):
    """What a model can do"""
    vision: bool
    audio: bool
    video: bool
    code: bool
    max_context_tokens: int
    cost_per_1k_tokens: float

class ModelInfo(TypedDict):
    """Selected model information"""
    name: str
    provider: Literal["ollama", "vllm", "openai-compatible"]
    endpoint: str
    
    capabilities: ModelCapabilities
    parameters: dict[str, Any]  # temperature, top_p, etc.
    
    # Routing metadata
    routing_reason: str  # "vision_required", "cost_optimized", "quality_required"
    fallback_model: Optional[str]

class ModelRoutingDecision(TypedDict):
    """Why this model was chosen"""
    primary_model: ModelInfo
    reasoning: str
    
    # For complex queries, we might use multiple models
    vision_model: Optional[ModelInfo]
    text_model: Optional[ModelInfo]

# ============================================================
# CHAT GRAPH STATE (SYNC - REASONING ONLY)
# ============================================================
class ChatGraphState(TypedDict):
    """
    State for real-time chat reasoning.
    
    RULES:
    - NEVER touches storage directly
    - ONLY reads pre-processed data
    - Fast execution (< 30 seconds)
    """
    
    # ---- Identity ----
    user_id: str
    session_id: str
    request_id: str
    timestamp: str
    
    # ---- User Input ----
    user_text: str
    
    # File references (NOT content, just metadata)
    attached_files: list[FileReference]
    
    # ---- Memory (pre-loaded) ----
    conversation_history: Annotated[list[BaseMessage], operator.add]
    
    # Retrieved from vector DB (already formatted for LLM)
    retrieved_context: list[RetrievedChunk]
    
    # ---- Model Selection ----
    routing_decision: Optional[ModelRoutingDecision]
    selected_model: Optional[ModelInfo]
    
    # ---- Context Assembly ----
    # This is what actually goes to the LLM
    llm_messages: Annotated[list[BaseMessage], operator.add]
    
    # Track token usage to avoid overflow
    current_token_count: int
    max_tokens_allowed: int
    
    # ---- Processing Status ----
    files_ready: bool  # All attached files processed?
    needs_retrieval: bool
    needs_vision_model: bool
    needs_multi_step: bool  # Complex agentic workflow?
    
    # ---- Output ----
    response_text: Optional[str]
    response_chunks: Annotated[list[str], operator.add]  # For streaming
    
    # Citations for transparency
    citations: list[dict[str, Any]]
    
    # ---- Artifacts to Generate ----
    # ChatGraph DECLARES intent, IngestionGraph executes
    pending_artifacts: list[dict[str, Any]]

    # ---- Tool Usage (Optional for Phase 2) ----
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    
    # ---- Error Handling ----
    errors: list[str]
    warnings: list[str]

# ============================================================
# INGESTION GRAPH STATE (ASYNC - STORAGE ONLY)
# ============================================================
class StorageDecision(TypedDict):
    """Deterministic storage routing (NO LLM calls here)"""
    store_in_s3: bool
    store_in_vector_db: bool
    store_in_postgres: bool
    
    reason: Literal[
        "too_large",           # > 50k tokens
        "multimedia",          # Images, video, audio
        "simple_text",         # < 5k tokens, no embedding needed
        "generated_artifact",  # AI-generated content
        "conversation_turn"    # Store as chat history
    ]

class IngestionGraphState(TypedDict):
    """
    State for async file processing.
    
    RULES:
    - NEVER does reasoning
    - ONLY does storage operations
    - Must be idempotent (can retry safely)
    """
    
    # ---- Identity ----
    user_id: str
    session_id: str
    file_id: str
    
    # ---- Source ----
    source_uri: str  # S3 URI
    mime_type: str
    size_bytes: int
    upload_timestamp: str
    
    # ---- Processing Pipeline ----
    processing_stage: Literal[
        "queued",
        "downloading",
        "extracting",
        "chunking",
        "embedding",
        "storing",
        "completed",
        "failed"
    ]
    
    # ---- Extracted Content ----
    # Text from documents
    extracted_text: Optional[str]
    
    # Video keyframes (S3 URIs)
    keyframe_uris: Optional[list[str]]
    
    # Audio transcript
    audio_transcript: Optional[str]
    
    # Image analysis results
    image_descriptions: Optional[list[str]]
    
    # ---- Chunking Results ----
    chunks: Optional[list[dict[str, Any]]]
    chunk_embeddings: Optional[list[list[float]]]
    
    # ---- Storage Decision ----
    storage_decision: Optional[StorageDecision]
    
    # ---- Persistence Results ----
    s3_uris: list[str]  # All S3 objects created
    vector_ids: list[str]  # Qdrant point IDs
    postgres_record_id: Optional[str]
    
    # ---- Processing Metadata ----
    processing_time_ms: Optional[int]
    retry_count: int
    
    # ---- Error Handling ----
    error_message: Optional[str]
    can_retry: bool

# ============================================================
# CONTEXT MANAGEMENT
# ============================================================
class ContextWindow(TypedDict):
    """Track what's in the LLM's context"""
    system_prompt_tokens: int
    history_tokens: int
    retrieved_context_tokens: int
    file_content_tokens: int
    user_query_tokens: int
    
    total_tokens: int
    remaining_tokens: int
    
    # What got truncated?
    truncated_history: bool
    truncated_retrieval: bool

class RetrievalStrategy(TypedDict):
    """How to retrieve relevant context"""
    method: Literal["vector_search", "keyword", "hybrid", "rerank"]
    
    # Vector search params
    top_k: int
    similarity_threshold: float
    
    # Filters
    file_types: Optional[list[str]]
    time_range: Optional[dict[str, str]]
    
    # Reranking
    use_reranker: bool
    reranker_model: Optional[str]

# ============================================================
# ROUTING & ORCHESTRATION
# ============================================================
class QueryAnalysis(TypedDict):
    """Initial analysis of user query to route correctly"""
    
    # Complexity
    complexity: Literal["simple", "medium", "complex"]
    estimated_tokens: int
    
    # Required capabilities
    needs_vision: bool
    needs_video_analysis: bool
    needs_audio: bool
    needs_code_execution: bool
    needs_web_search: bool
    
    # Multi-modal detection
    has_attached_files: bool
    file_types_attached: list[str]
    
    # Intent classification
    intent: Literal[
        "question_answering",
        "content_creation",
        "analysis",
        "summarization",
        "comparison",
        "code_generation",
        "other"
    ]
    
    # Should we wait for file processing?
    requires_file_processing: bool
    can_answer_immediately: bool

# ============================================================
# SOCIAL MEDIA SPECIFIC (FUTURE)
# ============================================================
class SocialContext(TypedDict):
    """
    Future: User's social graph context from Neo4j
    NOT needed for MVP
    """
    user_interests: list[str]
    connected_users: list[str]
    trending_topics: list[str]
    engagement_history: dict[str, Any]

# ============================================================
# MONITORING & OBSERVABILITY
# ============================================================
class ExecutionMetrics(TypedDict):
    """Track performance for optimization"""
    
    # Latency breakdown
    routing_time_ms: Optional[int]
    retrieval_time_ms: Optional[int]
    llm_inference_time_ms: Optional[int]
    total_time_ms: Optional[int]
    
    # Resource usage
    models_used: list[str]
    total_tokens: int
    estimated_cost_usd: float
    
    # Quality metrics
    retrieval_chunks_used: int
    context_utilization: float  # % of context window used
    
    # Errors
    retry_count: int
    error_count: int