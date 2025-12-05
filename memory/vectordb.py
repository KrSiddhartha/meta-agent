"""
Vector Database: Milvus-based semantic memory.

Collections:
- context_store: Conversation context and task history
- tool_registry: Persisted tools with semantic search
- agent_registry: Sub-agent configurations
"""

import os
import json
import hashlib
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class ContextEntry:
    """A context/memory entry."""
    content: str
    entry_type: str  # thought, action, observation, user_input
    task_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)


@dataclass
class ToolRecord:
    """A persisted tool record."""
    name: str
    description: str
    source_code: str
    capabilities: List[str]
    parameters: Dict
    example_usage: str = ""
    success_count: int = 0
    fail_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str = ""


@dataclass 
class AgentRecord:
    """A sub-agent configuration record."""
    name: str
    description: str
    system_prompt: str
    capabilities: List[str]
    tools_used: List[str] = field(default_factory=list)
    example_tasks: List[str] = field(default_factory=list)
    success_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class EmbeddingProvider:
    """Generate embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[Embeddings] Loaded {model_name} (dim={self.dim})")
    
    def embed(self, text: str) -> List[float]:
        """Embed single text."""
        return self.model.encode(text, convert_to_numpy=True).tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        return self.model.encode(texts, convert_to_numpy=True).tolist()


class MilvusVectorStore:
    """
    Milvus-based vector store for semantic memory.
    
    Supports both Milvus Lite (embedded) and standard Milvus server.
    """
    
    def __init__(
        self,
        db_path: str = "./agent_memory.db",
        embedding_model: str = "all-MiniLM-L6-v2",
        use_lite: bool = True
    ):
        """
        Initialize vector store.
        
        Args:
            db_path: Path for Milvus Lite DB or URI for server
            embedding_model: Sentence transformer model
            use_lite: Use Milvus Lite (embedded mode)
        """
        from pymilvus import MilvusClient
        
        self.embedder = EmbeddingProvider(embedding_model)
        self.dim = self.embedder.dim
        
        if use_lite:
            self.client = MilvusClient(db_path)
            print(f"[VectorDB] Milvus Lite: {db_path}")
        else:
            self.client = MilvusClient(uri=db_path)
            print(f"[VectorDB] Milvus Server: {db_path}")
        
        self._init_collections()
    
    def _init_collections(self):
        """Initialize required collections."""
        # Context store
        if not self.client.has_collection("context_store"):
            self.client.create_collection(
                collection_name="context_store",
                dimension=self.dim,
            )
            print("[VectorDB] Created: context_store")
        
        # Tool registry
        if not self.client.has_collection("tool_registry"):
            self.client.create_collection(
                collection_name="tool_registry",
                dimension=self.dim,
            )
            print("[VectorDB] Created: tool_registry")
        
        # Agent registry
        if not self.client.has_collection("agent_registry"):
            self.client.create_collection(
                collection_name="agent_registry",
                dimension=self.dim,
            )
            print("[VectorDB] Created: agent_registry")
    
    def _generate_id(self, *args) -> str:
        """Generate deterministic ID."""
        content = "".join(str(a) for a in args) + datetime.now().isoformat()
        return hashlib.md5(content.encode()).hexdigest()
    
    # ========================================================================
    # Context Operations
    # ========================================================================
    
    def store_context(self, entry: ContextEntry) -> str:
        """Store a context entry."""
        entry_id = self._generate_id(entry.content, entry.task_id)
        embedding = self.embedder.embed(entry.content)
        
        data = {
            "id": entry_id,
            "vector": embedding,
            "content": entry.content[:10000],
            "entry_type": entry.entry_type,
            "task_id": entry.task_id,
            "timestamp": entry.timestamp,
            "metadata": json.dumps(entry.metadata),
        }
        
        self.client.insert("context_store", [data])
        return entry_id
    
    def search_context(
        self,
        query: str,
        task_id: Optional[str] = None,
        limit: int = 10,
        score_threshold: float = 0.5
    ) -> List[Dict]:
        """Search context by semantic similarity."""
        query_embedding = self.embedder.embed(query)
        
        filter_expr = f'task_id == "{task_id}"' if task_id else None
        
        results = self.client.search(
            collection_name="context_store",
            data=[query_embedding],
            limit=limit,
            filter=filter_expr,
            output_fields=["content", "entry_type", "task_id", "timestamp"]
        )
        
        formatted = []
        for hits in results:
            for hit in hits:
                score = 1 - hit.get("distance", 1)  # Convert distance to similarity
                if score >= score_threshold:
                    entity = hit.get("entity", {})
                    formatted.append({
                        "content": entity.get("content"),
                        "entry_type": entity.get("entry_type"),
                        "task_id": entity.get("task_id"),
                        "timestamp": entity.get("timestamp"),
                        "score": score
                    })
        
        return formatted
    
    def get_task_context(self, task_id: str, limit: int = 20) -> List[Dict]:
        """Get all context for a specific task."""
        results = self.client.query(
            collection_name="context_store",
            filter=f'task_id == "{task_id}"',
            output_fields=["content", "entry_type", "timestamp"],
            limit=limit
        )
        
        # Sort by timestamp
        return sorted(results, key=lambda x: x.get("timestamp", ""))
    
    # ========================================================================
    # Tool Registry Operations
    # ========================================================================
    
    def register_tool(self, tool: ToolRecord) -> str:
        """Register a tool in the vector store."""
        tool_id = self._generate_id(tool.name)
        
        # Create rich text for embedding
        embed_text = f"{tool.name} {tool.description} {' '.join(tool.capabilities)}"
        embedding = self.embedder.embed(embed_text)
        
        data = {
            "id": tool_id,
            "vector": embedding,
            "name": tool.name,
            "description": tool.description[:2000],
            "source_code": tool.source_code[:30000],
            "capabilities": json.dumps(tool.capabilities),
            "parameters": json.dumps(tool.parameters),
            "example_usage": tool.example_usage[:1000],
            "success_count": tool.success_count,
            "fail_count": tool.fail_count,
            "created_at": tool.created_at,
            "last_used": tool.last_used,
        }
        
        self.client.insert("tool_registry", [data])
        print(f"[VectorDB] Registered tool: {tool.name}")
        return tool_id
    
    def search_tools(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.4
    ) -> List[Dict]:
        """Search for tools by capability/description."""
        query_embedding = self.embedder.embed(query)
        
        results = self.client.search(
            collection_name="tool_registry",
            data=[query_embedding],
            limit=limit,
            output_fields=["name", "description", "source_code", "capabilities", "parameters", "success_count"]
        )
        
        formatted = []
        for hits in results:
            for hit in hits:
                score = 1 - hit.get("distance", 1)
                if score >= score_threshold:
                    entity = hit.get("entity", {})
                    formatted.append({
                        "name": entity.get("name"),
                        "description": entity.get("description"),
                        "source_code": entity.get("source_code"),
                        "capabilities": json.loads(entity.get("capabilities", "[]")),
                        "parameters": json.loads(entity.get("parameters", "{}")),
                        "success_count": entity.get("success_count", 0),
                        "score": score
                    })
        
        return formatted
    
    def get_tool_by_name(self, name: str) -> Optional[Dict]:
        """Get a specific tool by name."""
        results = self.client.query(
            collection_name="tool_registry",
            filter=f'name == "{name}"',
            output_fields=["name", "description", "source_code", "capabilities", "parameters"],
            limit=1
        )
        
        if results:
            result = results[0]
            result["capabilities"] = json.loads(result.get("capabilities", "[]"))
            result["parameters"] = json.loads(result.get("parameters", "{}"))
            return result
        return None
    
    def update_tool_stats(self, name: str, success: bool):
        """Update tool usage statistics."""
        # Note: Milvus doesn't support updates well, so this is a placeholder
        # In production, you might use a separate stats store
        pass
    
    # ========================================================================
    # Agent Registry Operations
    # ========================================================================
    
    def register_agent(self, agent: AgentRecord) -> str:
        """Register a sub-agent configuration."""
        agent_id = self._generate_id(agent.name)
        
        embed_text = f"{agent.name} {agent.description} {' '.join(agent.capabilities)}"
        embedding = self.embedder.embed(embed_text)
        
        data = {
            "id": agent_id,
            "vector": embedding,
            "name": agent.name,
            "description": agent.description[:2000],
            "system_prompt": agent.system_prompt[:10000],
            "capabilities": json.dumps(agent.capabilities),
            "tools_used": json.dumps(agent.tools_used),
            "example_tasks": json.dumps(agent.example_tasks),
            "success_count": agent.success_count,
            "created_at": agent.created_at,
        }
        
        self.client.insert("agent_registry", [data])
        print(f"[VectorDB] Registered agent: {agent.name}")
        return agent_id
    
    def search_agents(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.4
    ) -> List[Dict]:
        """Search for agents by capability."""
        query_embedding = self.embedder.embed(query)
        
        results = self.client.search(
            collection_name="agent_registry",
            data=[query_embedding],
            limit=limit,
            output_fields=["name", "description", "system_prompt", "capabilities", "tools_used"]
        )
        
        formatted = []
        for hits in results:
            for hit in hits:
                score = 1 - hit.get("distance", 1)
                if score >= score_threshold:
                    entity = hit.get("entity", {})
                    formatted.append({
                        "name": entity.get("name"),
                        "description": entity.get("description"),
                        "system_prompt": entity.get("system_prompt"),
                        "capabilities": json.loads(entity.get("capabilities", "[]")),
                        "tools_used": json.loads(entity.get("tools_used", "[]")),
                        "score": score
                    })
        
        return formatted
    
    # ========================================================================
    # Utility
    # ========================================================================
    
    def get_stats(self) -> Dict:
        """Get store statistics."""
        return {
            "context_entries": self.client.get_collection_stats("context_store").get("row_count", 0),
            "tools": self.client.get_collection_stats("tool_registry").get("row_count", 0),
            "agents": self.client.get_collection_stats("agent_registry").get("row_count", 0),
        }
    
    def clear_collection(self, collection_name: str):
        """Clear all data from a collection."""
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
            self._init_collections()


# ============================================================================
# Factory Function
# ============================================================================

def create_vector_store(
    db_path: str = "./agent_memory.db",
    embedding_model: str = "all-MiniLM-L6-v2",
    use_lite: bool = True
) -> MilvusVectorStore:
    """Create a vector store instance."""
    return MilvusVectorStore(
        db_path=db_path,
        embedding_model=embedding_model,
        use_lite=use_lite
    )
