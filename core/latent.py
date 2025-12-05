"""
Latent Collaboration Module

Based on LatentMAS paper (arXiv:2511.20639):
- Agents communicate via KV-cache transfer instead of text
- Hidden states encode 235-471x more information than tokens
- Decode only at final agent (70-84% token savings, 4x speedup)

Key insight: Text serialization between agents is a massive information bottleneck.
KV cache preserves the full semantic richness of model reasoning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


# Language-specific configurations based on scaling law research
# Higher α languages (Python) need more latent steps; lower α (Rust) saturate faster
LANGUAGE_LATENT_CONFIG = {
    "python": {
        "latent_steps": 60,        # High α=0.221, benefits from more reasoning
        "temperature": 0.7,
        "description": "Dynamic typing requires more exploration"
    },
    "rust": {
        "latent_steps": 35,        # Lower α=0.643, saturates faster
        "temperature": 0.5,
        "description": "Strict type system, more predictable"
    },
    "javascript": {
        "latent_steps": 50,
        "temperature": 0.6,
        "description": "Dynamic but more constrained than Python"
    },
    "typescript": {
        "latent_steps": 45,
        "temperature": 0.55,
        "description": "Type hints reduce ambiguity"
    },
    "go": {
        "latent_steps": 40,
        "temperature": 0.5,
        "description": "Simple, explicit, low complexity"
    },
    "java": {
        "latent_steps": 45,
        "temperature": 0.5,
        "description": "Verbose but predictable"
    },
    "bash": {
        "latent_steps": 30,
        "temperature": 0.6,
        "description": "Short scripts, less reasoning needed"
    },
    "default": {
        "latent_steps": 50,
        "temperature": 0.6,
        "description": "Balanced default"
    }
}


@dataclass
class LatentThoughts:
    """Container for latent reasoning output."""
    hidden_states: torch.Tensor          # Shape: [num_steps, hidden_dim]
    working_memory: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]  # KV cache
    num_steps: int
    language: str = "python"
    
    def get_final_hidden(self) -> torch.Tensor:
        """Get the last hidden state for continuation."""
        return self.hidden_states[-1]


@dataclass 
class LatentTransferResult:
    """Result of transferring latent memory between agents."""
    success: bool
    combined_kv: Optional[Tuple] = None
    error: Optional[str] = None


class InputOutputAligner:
    """
    Computes alignment matrix W_a to map hidden states back to input space.
    
    From LatentMAS Equation 3:
        e = h @ W_a, where W_a ≈ W_out^(-1) @ W_in
    
    This prevents distribution drift when feeding hidden states as input.
    Uses ridge regression for numerical stability.
    """
    
    def __init__(self, model, lambda_reg: float = 1e-4):
        self.model = model
        self.lambda_reg = lambda_reg
        self.W_a = None
        self.device = None
        self._computed = False
    
    def compute(self) -> torch.Tensor:
        """Compute alignment matrix once, reuse for all latent steps."""
        if self._computed:
            return self.W_a
        
        # Get embedding matrices
        W_in = self._get_input_embeddings()
        W_out = self._get_output_embeddings()
        
        self.device = W_in.device
        
        # Ridge regression: W_a = (W_out^T @ W_out + λI)^(-1) @ W_out^T @ W_in
        # This is more stable than direct pseudo-inverse
        d_h = W_out.shape[1]
        
        WtW = W_out.T @ W_out
        regularizer = self.lambda_reg * torch.eye(d_h, device=self.device)
        
        self.W_a = torch.linalg.solve(
            WtW + regularizer,
            W_out.T @ W_in
        )
        
        self._computed = True
        return self.W_a
    
    def align(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Map hidden state to valid input embedding space."""
        if not self._computed:
            self.compute()
        return hidden_state @ self.W_a
    
    def _get_input_embeddings(self) -> torch.Tensor:
        """Extract input embedding matrix from model."""
        # Try common attribute names
        for attr in ['get_input_embeddings', 'embed_tokens', 'wte']:
            if hasattr(self.model, attr):
                emb = getattr(self.model, attr)
                if callable(emb):
                    emb = emb()
                if hasattr(emb, 'weight'):
                    return emb.weight.data
        raise AttributeError("Could not find input embeddings in model")
    
    def _get_output_embeddings(self) -> torch.Tensor:
        """Extract output embedding matrix (LM head) from model."""
        # Try common attribute names
        for attr in ['lm_head', 'get_output_embeddings', 'embed_out']:
            if hasattr(self.model, attr):
                head = getattr(self.model, attr)
                if callable(head):
                    head = head()
                if hasattr(head, 'weight'):
                    return head.weight.data
        raise AttributeError("Could not find output embeddings in model")


class LatentThoughtGenerator:
    """
    Generates latent thoughts via auto-regressive hidden state generation.
    
    Instead of decoding tokens at each step, we:
    1. Get last-layer hidden state
    2. Align it to input space via W_a
    3. Feed aligned embedding as next input
    4. Repeat for m latent steps
    
    Result: Rich continuous representations without text bottleneck.
    """
    
    def __init__(
        self, 
        model,
        tokenizer=None,
        default_steps: int = 50,
        lambda_reg: float = 1e-4
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.default_steps = default_steps
        self.aligner = InputOutputAligner(model, lambda_reg)
        
        # Compute alignment matrix once
        self.aligner.compute()
    
    def generate(
        self,
        input_ids: torch.Tensor,
        num_steps: Optional[int] = None,
        language: str = "python",
        attention_mask: Optional[torch.Tensor] = None,
        prefix_kv: Optional[Tuple] = None,
    ) -> LatentThoughts:
        """
        Generate latent thoughts from input.
        
        Args:
            input_ids: Tokenized input [batch, seq_len]
            num_steps: Number of latent reasoning steps (auto if None)
            language: Programming language for optimal step count
            attention_mask: Optional attention mask
            prefix_kv: KV cache from previous agent (for latent transfer)
        
        Returns:
            LatentThoughts with hidden states and working memory
        """
        # Get language-specific config
        if num_steps is None:
            config = LANGUAGE_LATENT_CONFIG.get(language, LANGUAGE_LATENT_CONFIG["default"])
            num_steps = config["latent_steps"]
        
        # Ensure model is in eval mode
        self.model.eval()
        
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Initial forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=True,
                past_key_values=prefix_kv,
            )
        
        # Collect hidden states
        hidden_states = []
        
        # Get last hidden state at final position
        # Shape: [batch, hidden_dim]
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        hidden_states.append(last_hidden)
        
        # Get KV cache
        past_kv = outputs.past_key_values
        
        # Auto-regressive latent generation
        for step in range(num_steps):
            # Align hidden state to input embedding space
            aligned_embedding = self.aligner.align(last_hidden)
            
            # Forward with aligned embedding as input
            with torch.no_grad():
                outputs = self.model(
                    inputs_embeds=aligned_embedding.unsqueeze(1),  # [batch, 1, hidden]
                    past_key_values=past_kv,
                    output_hidden_states=True,
                    use_cache=True,
                )
            
            # Extract new hidden state
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            hidden_states.append(last_hidden)
            
            # Update KV cache
            past_kv = outputs.past_key_values
        
        # Stack hidden states: [num_steps+1, batch, hidden_dim]
        hidden_stack = torch.stack(hidden_states, dim=0)
        
        return LatentThoughts(
            hidden_states=hidden_stack,
            working_memory=past_kv,
            num_steps=num_steps,
            language=language
        )
    
    def decode_from_latent(
        self,
        latent_thoughts: LatentThoughts,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Decode text from latent thoughts (only call at final agent!).
        
        This should be used ONLY by the final agent in the pipeline.
        All intermediate agents should pass KV caches, not text.
        """
        # Get final hidden state
        final_hidden = latent_thoughts.get_final_hidden()
        
        # Use it as input to generate text
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=final_hidden.unsqueeze(1),
                past_key_values=latent_thoughts.working_memory,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else 0,
            )
        
        return outputs


class WorkingMemoryTransfer:
    """
    Handles KV cache transfer between agents.
    
    From LatentMAS Section 3.2:
    - Working memory M_A = {K_cache, V_cache} for all L layers
    - Successive agent prepends predecessor's KV to its own
    - This preserves BOTH input context AND generated latent thoughts
    
    Key insight: This is lossless information transfer (Theorem 3.3).
    """
    
    @staticmethod
    def extract_working_memory(
        latent_thoughts: LatentThoughts
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """Extract KV cache as working memory."""
        return latent_thoughts.working_memory
    
    @staticmethod
    def transfer(
        source_kv: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        target_kv: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
    ) -> LatentTransferResult:
        """
        Transfer working memory from source agent to target agent.
        
        Performs layer-wise KV concatenation:
            K_target = [K_source; K_target]
            V_target = [V_source; V_target]
        """
        try:
            if target_kv is None:
                # No target KV, just use source
                return LatentTransferResult(success=True, combined_kv=source_kv)
            
            # Layer-wise concatenation
            combined_layers = []
            
            for layer_idx, (source_layer, target_layer) in enumerate(zip(source_kv, target_kv)):
                source_k, source_v = source_layer
                target_k, target_v = target_layer
                
                # Concatenate along sequence dimension (dim=2 for [batch, heads, seq, dim])
                combined_k = torch.cat([source_k, target_k], dim=2)
                combined_v = torch.cat([source_v, target_v], dim=2)
                
                combined_layers.append((combined_k, combined_v))
            
            return LatentTransferResult(
                success=True,
                combined_kv=tuple(combined_layers)
            )
        
        except Exception as e:
            return LatentTransferResult(
                success=False,
                error=f"KV transfer failed: {type(e).__name__}: {e}"
            )
    
    @staticmethod
    def validate_compatibility(
        kv1: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        kv2: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    ) -> Tuple[bool, str]:
        """Check if two KV caches are compatible for merging."""
        
        if len(kv1) != len(kv2):
            return False, f"Layer count mismatch: {len(kv1)} vs {len(kv2)}"
        
        for i, ((k1, v1), (k2, v2)) in enumerate(zip(kv1, kv2)):
            # Check dimensions (should match except sequence length)
            if k1.shape[0] != k2.shape[0]:  # batch
                return False, f"Layer {i}: batch size mismatch"
            if k1.shape[1] != k2.shape[1]:  # heads
                return False, f"Layer {i}: head count mismatch"
            if k1.shape[3] != k2.shape[3]:  # head_dim
                return False, f"Layer {i}: head dimension mismatch"
        
        return True, "Compatible"


class LatentAgent:
    """
    A single agent capable of latent reasoning and communication.
    
    Combines:
    - Latent thought generation (hidden states)
    - Working memory (KV cache)
    - Input-output alignment
    - Language-aware configuration
    """
    
    def __init__(
        self,
        name: str,
        model,
        tokenizer,
        role: str = "general",
        languages: List[str] = None,
    ):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.role = role
        self.languages = languages or ["python"]
        
        # Initialize latent generator
        self.generator = LatentThoughtGenerator(model, tokenizer)
        
        # Received working memory from other agents
        self.received_memory: Optional[Tuple] = None
        
        # Own latent thoughts
        self.latent_thoughts: Optional[LatentThoughts] = None
    
    def receive_working_memory(self, working_memory: Tuple):
        """Receive KV cache from another agent."""
        self.received_memory = working_memory
    
    def think(
        self,
        prompt: str,
        language: str = None,
        num_steps: int = None,
    ) -> LatentThoughts:
        """
        Generate latent thoughts for a prompt.
        
        Uses received working memory if available.
        """
        if language is None:
            language = self.languages[0]
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        device = next(self.model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Generate latent thoughts
        self.latent_thoughts = self.generator.generate(
            input_ids=input_ids,
            num_steps=num_steps,
            language=language,
            attention_mask=attention_mask,
            prefix_kv=self.received_memory,
        )
        
        return self.latent_thoughts
    
    def get_working_memory(self) -> Optional[Tuple]:
        """Get this agent's working memory for transfer."""
        if self.latent_thoughts is None:
            return None
        return WorkingMemoryTransfer.extract_working_memory(self.latent_thoughts)
    
    def decode(
        self,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
    ) -> str:
        """
        Decode latent thoughts to text.
        
        WARNING: This should only be called by the FINAL agent!
        Intermediate agents should transfer working memory instead.
        """
        if self.latent_thoughts is None:
            return ""
        
        output_ids = self.generator.decode_from_latent(
            self.latent_thoughts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    def reset(self):
        """Clear received memory and latent thoughts for new task."""
        self.received_memory = None
        self.latent_thoughts = None


# ============================================================================
# Pipeline Orchestrator
# ============================================================================

class LatentPipeline:
    """
    Orchestrates latent collaboration between multiple agents.
    
    Sequential Pipeline:
        Planner -> Critic -> Coder -> Solver
        
    Each agent receives predecessor's KV cache.
    Only Solver decodes to text.
    """
    
    def __init__(self, agents: Dict[str, LatentAgent]):
        self.agents = agents
        self.pipeline_order: List[str] = []
    
    def set_pipeline(self, order: List[str]):
        """Set the order of agents in the pipeline."""
        self.pipeline_order = order
    
    def run(
        self,
        task: str,
        language: str = "python",
        final_agent: str = None,
    ) -> str:
        """
        Run the latent collaboration pipeline.
        
        All agents except the final one only generate latent thoughts.
        Final agent decodes to text.
        """
        if not self.pipeline_order:
            raise ValueError("Pipeline order not set. Call set_pipeline() first.")
        
        if final_agent is None:
            final_agent = self.pipeline_order[-1]
        
        # Reset all agents
        for agent in self.agents.values():
            agent.reset()
        
        current_memory = None
        
        for i, agent_name in enumerate(self.pipeline_order):
            agent = self.agents.get(agent_name)
            if agent is None:
                raise ValueError(f"Agent '{agent_name}' not found")
            
            # Receive memory from previous agent
            if current_memory is not None:
                agent.receive_working_memory(current_memory)
            
            # Build prompt based on role
            if i == 0:
                prompt = f"Task: {task}\n\nPlan the approach:"
            elif agent.role == "critic":
                prompt = "Review and critique the plan:"
            elif agent.role == "coder":
                prompt = f"Write {language} code to solve the task:"
            else:
                prompt = "Synthesize and provide the final answer:"
            
            # Generate latent thoughts
            agent.think(prompt, language=language)
            
            # Extract working memory for next agent
            current_memory = agent.get_working_memory()
            
            # If this is the final agent, decode
            if agent_name == final_agent:
                return agent.decode()
        
        return ""


# ============================================================================
# Factory Functions
# ============================================================================

def create_latent_agent(
    name: str,
    model,
    tokenizer,
    role: str = "general",
    languages: List[str] = None,
) -> LatentAgent:
    """Create a latent-capable agent."""
    return LatentAgent(
        name=name,
        model=model,
        tokenizer=tokenizer,
        role=role,
        languages=languages,
    )


def create_sequential_pipeline(
    model,
    tokenizer,
) -> LatentPipeline:
    """
    Create a standard sequential pipeline.
    
    Planner -> Critic -> Refiner -> Solver
    
    Based on LatentMAS paper's chain-of-agents design.
    """
    agents = {
        "planner": create_latent_agent("planner", model, tokenizer, role="planner"),
        "critic": create_latent_agent("critic", model, tokenizer, role="critic"),
        "refiner": create_latent_agent("refiner", model, tokenizer, role="refiner"),
        "solver": create_latent_agent("solver", model, tokenizer, role="solver"),
    }
    
    pipeline = LatentPipeline(agents)
    pipeline.set_pipeline(["planner", "critic", "refiner", "solver"])
    
    return pipeline


def create_hierarchical_pipeline(
    model,
    tokenizer,
    specialist_languages: List[str] = None,
) -> LatentPipeline:
    """
    Create a hierarchical pipeline with specialists.
    
    Python Agent \
    Rust Agent   -> Summarizer -> Final Output
    Data Agent  /
    
    Based on LatentMAS paper's domain-specialized design.
    """
    specialist_languages = specialist_languages or ["python", "rust", "javascript"]
    
    agents = {}
    
    # Create specialist agents
    for lang in specialist_languages:
        agents[f"{lang}_specialist"] = create_latent_agent(
            f"{lang}_specialist",
            model,
            tokenizer,
            role="specialist",
            languages=[lang],
        )
    
    # Create summarizer
    agents["summarizer"] = create_latent_agent(
        "summarizer",
        model,
        tokenizer,
        role="summarizer",
        languages=specialist_languages,
    )
    
    pipeline = LatentPipeline(agents)
    
    # For hierarchical, all specialists run, then summarizer aggregates
    # (Note: Full hierarchical would need parallel execution, this is sequential)
    order = [f"{lang}_specialist" for lang in specialist_languages] + ["summarizer"]
    pipeline.set_pipeline(order)
    
    return pipeline
