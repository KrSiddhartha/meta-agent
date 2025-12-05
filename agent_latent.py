"""
Meta-Agent v2.2: Latent Collaboration Edition

Combines the self-extending meta-agent with latent communication from LatentMAS.

Key improvements over v2.1:
- Sub-agents communicate via KV-cache instead of text (70-84% token savings)
- Language-aware latent step configuration (Python needs more, Rust less)
- Decode only at final agent (4x speedup)
- Falls back to text mode when latent not available

Usage:
    agent = create_latent_meta_agent(model, tokenizer)
    result = agent.run("Create a fast fibonacci in Rust")
"""

import os
import uuid
import json
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Core
from core.llm import LLMProvider, create_llm
from core.parser import ResponseParser
from core.tools import Tool, ToolRegistry

# Latent collaboration (optional - requires torch)
try:
    from core.latent import (
        LatentAgent,
        LatentPipeline,
        LatentThoughts,
        WorkingMemoryTransfer,
        LANGUAGE_LATENT_CONFIG,
        create_latent_agent,
    )
    LATENT_IMPORTS_AVAILABLE = True
except ImportError:
    LATENT_IMPORTS_AVAILABLE = False
    # Placeholders for type hints
    LatentAgent = None
    LatentPipeline = None
    LatentThoughts = None
    WorkingMemoryTransfer = None
    LANGUAGE_LATENT_CONFIG = {
        "python": {"latent_steps": 60, "temperature": 0.7},
        "rust": {"latent_steps": 35, "temperature": 0.5},
        "default": {"latent_steps": 50, "temperature": 0.6},
    }
    create_latent_agent = None

# Sandbox
from sandbox.executor import (
    UnifiedExecutor, ExecutorConfig, ExecutionResult,
    PermissionLevel, create_executor
)

# Memory
from memory.script_manager import ScriptManager


class CollaborationMode(Enum):
    """How agents communicate."""
    TEXT = "text"          # Traditional text-based (verbose but compatible)
    LATENT = "latent"      # KV-cache transfer (efficient but requires same model)
    HYBRID = "hybrid"      # Latent for reasoning, text for final output


@dataclass
class LatentAgentConfig:
    """Configuration for latent-enabled agent."""
    storage_path: str = "./agent_data"
    max_iterations: int = 15
    verbose: bool = True
    
    # Latent collaboration settings
    collaboration_mode: CollaborationMode = CollaborationMode.HYBRID
    default_latent_steps: int = 50
    
    # Execution settings
    timeout_seconds: int = 60
    memory_limit: str = "512m"
    
    # Language defaults
    default_language: str = "python"


@dataclass
class SubAgentSpec:
    """Specification for a sub-agent."""
    name: str
    role: str
    description: str
    languages: List[str] = field(default_factory=lambda: ["python"])
    permission: PermissionLevel = PermissionLevel.RESTRICTED
    latent_steps: Optional[int] = None  # None = auto based on language


class LatentMetaAgent:
    """
    Self-extending meta-agent with latent collaboration.
    
    When collaboration_mode is LATENT or HYBRID:
    - Sub-agents share KV caches instead of text
    - Only final agent decodes to text
    - 70-84% fewer tokens, 4x faster
    
    When collaboration_mode is TEXT:
    - Falls back to traditional text exchange
    - More verbose but works across different models
    """
    
    SYSTEM_PROMPT = """You are a Meta-Agent with latent collaboration capability.

## Available Actions:

1. **use_tool** - Use an existing tool
   ACTION_INPUT: {{"tool_name": "...", "parameters": {{...}}}}

2. **create_tool** - Create a new tool
   ACTION_INPUT: {{"name": "...", "description": "...", "code": "...", "language": "python|rust|bash"}}

3. **execute_code** - Execute code directly
   ACTION_INPUT: {{"code": "...", "language": "python|rust|bash", "permission": "restricted|elevated"}}

4. **search_tools** - Search for existing tools
   ACTION_INPUT: {{"query": "..."}}

5. **latent_reason** - Use latent sub-agent pipeline for complex reasoning
   ACTION_INPUT: {{"task": "...", "language": "python|rust|go|javascript", "pipeline": "sequential|hierarchical"}}

6. **final_answer** - Return the final answer
   ACTION_INPUT: {{"answer": "..."}}

## Collaboration Mode: {collaboration_mode}
{mode_description}

## Available Tools:
{tools_description}

## Response Format:
THOUGHT: [Your reasoning]

ACTION: [action_name]

ACTION_INPUT:
```json
{{...}}
```

## Guidelines:
1. Use latent_reason for complex multi-step reasoning tasks
2. Choose language based on task requirements
3. Sequential pipeline: Planner -> Critic -> Refiner -> Solver
4. For simple tasks, execute_code directly is faster

Begin!
"""

    MODE_DESCRIPTIONS = {
        CollaborationMode.TEXT: "Agents communicate via natural language text.",
        CollaborationMode.LATENT: "Agents communicate via KV-cache (efficient, same model only).",
        CollaborationMode.HYBRID: "Latent reasoning internally, text output externally.",
    }

    def __init__(
        self,
        llm: LLMProvider,
        model=None,
        tokenizer=None,
        config: LatentAgentConfig = None,
    ):
        self.llm = llm
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or LatentAgentConfig()
        
        os.makedirs(self.config.storage_path, exist_ok=True)
        
        self._init_components()
        
        self._log(f"Meta-Agent v2.2 (Latent) initialized")
        self._log(f"Collaboration mode: {self.config.collaboration_mode.value}")
        self._log(f"Latent available: {self._latent_available}")
    
    def _init_components(self):
        """Initialize all components."""
        self.parser = ResponseParser()
        self.tools = ToolRegistry()
        
        # Executor
        exec_config = ExecutorConfig(
            timeout_seconds=self.config.timeout_seconds,
            memory_limit=self.config.memory_limit,
        )
        self.executor = create_executor(exec_config)
        
        # Script manager
        self.script_manager = ScriptManager(
            base_path=f"{self.config.storage_path}/scripts"
        )
        
        # Load persisted tools
        self._load_persisted_tools()
        
        # Initialize latent pipeline if model available
        self._latent_available = False
        self.latent_pipeline = None
        
        if self.model is not None and self.tokenizer is not None:
            try:
                self._init_latent_pipeline()
                self._latent_available = True
            except Exception as e:
                self._log(f"Latent pipeline init failed: {e}")
                self._log("Falling back to text-only mode")
    
    def _init_latent_pipeline(self):
        """Initialize the latent collaboration pipeline."""
        if not LATENT_IMPORTS_AVAILABLE:
            raise ImportError("Latent imports not available (torch not installed)")
        
        # Create sub-agents for sequential pipeline
        agents = {
            "planner": create_latent_agent(
                "planner", self.model, self.tokenizer,
                role="planner", languages=["python", "rust", "go"]
            ),
            "critic": create_latent_agent(
                "critic", self.model, self.tokenizer,
                role="critic", languages=["python", "rust", "go"]
            ),
            "refiner": create_latent_agent(
                "refiner", self.model, self.tokenizer,
                role="refiner", languages=["python", "rust", "go"]
            ),
            "solver": create_latent_agent(
                "solver", self.model, self.tokenizer,
                role="solver", languages=["python", "rust", "go"]
            ),
        }
        
        self.latent_pipeline = LatentPipeline(agents)
        self.latent_pipeline.set_pipeline(["planner", "critic", "refiner", "solver"])
    
    def _load_persisted_tools(self):
        """Load persisted tools."""
        tool_names = self.script_manager.list_tools()
        loaded = 0
        
        for name in tool_names:
            info = self.script_manager.get_tool_info(name)
            func = self.script_manager.load_tool(name)
            
            if func and info:
                tool = Tool(
                    name=name,
                    description=info.get("description", ""),
                    function=func,
                    parameters=info.get("parameters", {}),
                )
                self.tools.register(tool)
                loaded += 1
        
        if loaded:
            self._log(f"Loaded {loaded} persisted tools")
    
    def _log(self, message: str):
        if self.config.verbose:
            print(f"[LatentAgent] {message}")
    
    def _build_prompt(self, user_input: str) -> str:
        mode_desc = self.MODE_DESCRIPTIONS.get(
            self.config.collaboration_mode,
            "Unknown mode"
        )
        
        if not self._latent_available:
            mode_desc += " (Latent not available - using text fallback)"
        
        system = self.SYSTEM_PROMPT.format(
            collaboration_mode=self.config.collaboration_mode.value,
            mode_description=mode_desc,
            tools_description=self.tools.get_tools_description(),
        )
        
        return f"{system}\n\nUser Request: {user_input}\n\nTHOUGHT:"
    
    # ========================================================================
    # Action Handlers
    # ========================================================================
    
    def _execute_action(self, action: str, action_input: Dict) -> str:
        handlers = {
            "use_tool": self._action_use_tool,
            "create_tool": self._action_create_tool,
            "execute_code": self._action_execute_code,
            "search_tools": self._action_search_tools,
            "latent_reason": self._action_latent_reason,
            "final_answer": self._action_final_answer,
        }
        
        handler = handlers.get(action)
        if not handler:
            return f"Unknown action: {action}. Valid: {list(handlers.keys())}"
        
        try:
            return handler(action_input)
        except Exception as e:
            return f"Action error: {type(e).__name__}: {e}"
    
    def _action_use_tool(self, params: Dict) -> str:
        tool_name = params.get("tool_name", params.get("name", ""))
        tool_params = params.get("parameters", {})
        
        if not tool_name:
            return "Error: No tool_name specified"
        
        return self.tools.execute(tool_name, **tool_params)
    
    def _action_create_tool(self, params: Dict) -> str:
        name = params.get("name", "")
        description = params.get("description", "")
        code = params.get("code", "")
        language = params.get("language", "python")
        tool_params = params.get("parameters", {})
        
        if not name or not code:
            return "Error: Tool requires 'name' and 'code'"
        
        # Test code first
        self._log(f"Testing tool: {name} ({language})")
        result = self.executor.execute(code, language=language)
        
        if not result.success:
            return f"Tool test failed: {result.error}"
        
        # Create in registry (only Python tools can be registered as functions)
        if language == "python":
            tool = self.tools.create_tool_from_code(name, description, code, tool_params)
            if not tool:
                return f"Failed to create tool '{name}'"
        
        # Persist
        self.script_manager.save_tool(name, description, code, tool_params)
        
        return f"Created and persisted tool '{name}' ({language})"
    
    def _action_execute_code(self, params: Dict) -> str:
        code = params.get("code", "")
        language = params.get("language", "python").lower()
        permission_str = params.get("permission", "restricted").lower()
        
        if not code:
            return "Error: No code provided"
        
        try:
            permission = PermissionLevel(permission_str)
        except ValueError:
            permission = PermissionLevel.RESTRICTED
        
        # Language-specific options
        kwargs = {}
        if language == "rust":
            kwargs["cargo_deps"] = params.get("cargo_deps", {})
        elif language == "python":
            kwargs["inputs"] = params.get("inputs", {})
            kwargs["packages"] = params.get("packages")
        
        self._log(f"Executing {language} code (permission: {permission.value})")
        result = self.executor.execute(code, language=language, permission=permission, **kwargs)
        
        if result.success:
            output = result.output or "Executed successfully"
            if result.return_value is not None:
                output += f"\nReturn: {result.return_value}"
            return output
        else:
            return f"Execution failed: {result.error}"
    
    def _action_search_tools(self, params: Dict) -> str:
        query = params.get("query", "")
        if not query:
            return "Error: No search query"
        
        results = self.script_manager.search_tools(query, limit=5)
        
        if not results:
            return f"No tools found for: '{query}'"
        
        output = [f"Found {len(results)} tools:"]
        for r in results:
            output.append(f"• {r['name']}: {r['description'][:50]}...")
        return "\n".join(output)
    
    def _action_latent_reason(self, params: Dict) -> str:
        """
        Use latent sub-agent pipeline for complex reasoning.
        
        This is the key action that leverages LatentMAS:
        - All sub-agents communicate via KV-cache
        - Only final agent decodes to text
        - 70-84% fewer tokens, 4x faster
        """
        task = params.get("task", "")
        language = params.get("language", self.config.default_language)
        pipeline_type = params.get("pipeline", "sequential")
        
        if not task:
            return "Error: No task specified for latent reasoning"
        
        # Check if latent is available
        if not self._latent_available:
            self._log("Latent not available, using text fallback")
            return self._text_reasoning_fallback(task, language)
        
        # Get language-specific latent config
        lang_config = LANGUAGE_LATENT_CONFIG.get(language, LANGUAGE_LATENT_CONFIG["default"])
        self._log(f"Latent reasoning: {language} ({lang_config['latent_steps']} steps)")
        
        try:
            # Run latent pipeline
            result = self.latent_pipeline.run(
                task=task,
                language=language,
            )
            
            return f"[Latent Result]\n{result}"
        
        except Exception as e:
            self._log(f"Latent reasoning failed: {e}")
            return self._text_reasoning_fallback(task, language)
    
    def _text_reasoning_fallback(self, task: str, language: str) -> str:
        """Fallback to text-based reasoning when latent not available."""
        # Simple text-based chain of thought
        prompt = f"""Task: {task}

Think step by step:
1. Understand the problem
2. Plan the approach
3. Consider edge cases
4. Provide solution in {language}

Solution:"""
        
        response = self.llm.generate(prompt, max_tokens=1024)
        return f"[Text Reasoning]\n{response}"
    
    def _action_final_answer(self, params: Dict) -> str:
        answer = params.get("answer", str(params))
        return f"FINAL_ANSWER: {answer}"
    
    # ========================================================================
    # Main Run Loop
    # ========================================================================
    
    def run(self, user_input: str) -> str:
        """Run the agent on a user request."""
        
        self._log(f"\n{'='*60}")
        self._log(f"Task: {user_input[:80]}...")
        self._log(f"{'='*60}")
        
        full_prompt = self._build_prompt(user_input)
        final_answer = None
        
        for iteration in range(self.config.max_iterations):
            self._log(f"\n--- Step {iteration + 1} ---")
            
            response = self.llm.generate(full_prompt, max_tokens=2048)
            self._log(f"Response: {response[:200]}...")
            
            parsed = self.parser.parse(response)
            
            if parsed.thought:
                self._log(f"💭 {parsed.thought[:100]}...")
            
            if not parsed.action:
                final_answer = parsed.thought or response
                break
            
            self._log(f"⚡ {parsed.action}")
            
            result = self._execute_action(parsed.action, parsed.action_input)
            self._log(f"📊 {result[:150]}...")
            
            if result.startswith("FINAL_ANSWER:"):
                final_answer = result[13:].strip()
                break
            
            full_prompt += f" {parsed.thought}\n\nACTION: {parsed.action}\n\nACTION_INPUT:\n```json\n{json.dumps(parsed.action_input, indent=2)}\n```\n\nOBSERVATION: {result}\n\nTHOUGHT:"
        
        else:
            final_answer = f"Max iterations reached. Last: {response[:200]}"
        
        self._log(f"\n{'='*60}")
        self._log(f"Completed")
        self._log(f"{'='*60}")
        
        return final_answer
    
    def run_latent(self, task: str, language: str = "python") -> str:
        """
        Direct latent reasoning without the ReAct loop.
        
        For simple tasks where you just want latent sub-agent collaboration.
        """
        if not self._latent_available:
            return self._text_reasoning_fallback(task, language)
        
        return self.latent_pipeline.run(task=task, language=language)


# ============================================================================
# Factory Functions
# ============================================================================

def create_latent_meta_agent(
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    provider: str = "huggingface",
    collaboration_mode: CollaborationMode = CollaborationMode.HYBRID,
    storage_path: str = "./agent_data",
    verbose: bool = True,
    **llm_kwargs
) -> LatentMetaAgent:
    """
    Create a latent-enabled meta-agent.
    
    For full latent capability, you need to provide model and tokenizer.
    Otherwise, falls back to text-only mode.
    """
    llm = create_llm(model_name, provider, **llm_kwargs)
    
    config = LatentAgentConfig(
        storage_path=storage_path,
        verbose=verbose,
        collaboration_mode=collaboration_mode,
    )
    
    # Try to load model and tokenizer for latent mode
    model = None
    tokenizer = None
    
    if collaboration_mode in [CollaborationMode.LATENT, CollaborationMode.HYBRID]:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            if verbose:
                print(f"[Factory] Loading model for latent mode: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            
            if verbose:
                print(f"[Factory] Model loaded successfully")
        
        except Exception as e:
            if verbose:
                print(f"[Factory] Could not load model for latent: {e}")
                print(f"[Factory] Falling back to text-only mode")
    
    return LatentMetaAgent(
        llm=llm,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Meta-Agent v2.2 (Latent)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--provider", choices=["huggingface", "local", "vllm"], default="huggingface")
    parser.add_argument("--storage", default="./agent_data")
    parser.add_argument("--mode", choices=["text", "latent", "hybrid"], default="hybrid")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║              Meta-Agent v2.2 (Latent Edition)                 ║
║                                                               ║
║  • Latent collaboration via KV-cache transfer                ║
║  • 70-84% fewer tokens, 4x faster                            ║
║  • Language-aware latent step configuration                  ║
║  • Falls back to text when latent unavailable                ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    mode = CollaborationMode(args.mode)
    
    llm_kwargs = {}
    if args.provider == "vllm":
        llm_kwargs["base_url"] = args.vllm_url
    
    agent = create_latent_meta_agent(
        model_name=args.model,
        provider=args.provider,
        collaboration_mode=mode,
        storage_path=args.storage,
        verbose=not args.quiet,
        **llm_kwargs
    )
    
    print(f"\nReady! Type 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                break
            
            response = agent.run(user_input)
            print(f"\n🤖 {response}\n")
        
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
