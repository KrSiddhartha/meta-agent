"""
Meta-Agent v2.1: Self-Extending with Sub-Agents

New capabilities:
- create_agent: Create specialized sub-agents
- spawn_agent: Delegate tasks to sub-agents
- search_agents: Find existing agents by capability
- Multi-language: Python, Rust, Bash
- Permission levels: RESTRICTED, ELEVATED, SYSTEM
"""

import os
import uuid
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

# Core
from core.llm import LLMProvider, create_llm
from core.parser import ResponseParser, ParsedResponse
from core.tools import Tool, ToolRegistry

# Sandbox - unified executor
from sandbox.executor import (
    UnifiedExecutor, ExecutorConfig, ExecutionResult,
    PermissionLevel, create_executor
)

# Memory
from memory.vectordb import (
    MilvusVectorStore, ContextEntry, ToolRecord, AgentRecord,
    create_vector_store
)
from memory.script_manager import ScriptManager


@dataclass
class AgentConfig:
    """Configuration for the meta-agent."""
    storage_path: str = "./agent_data"
    max_iterations: int = 15
    verbose: bool = True
    
    # Default permission for created tools/agents
    default_permission: PermissionLevel = PermissionLevel.RESTRICTED
    
    # Execution settings
    timeout_seconds: int = 60
    memory_limit: str = "512m"


@dataclass
class SubAgent:
    """A sub-agent that can be spawned."""
    name: str
    description: str
    system_prompt: str
    capabilities: List[str]
    permission: PermissionLevel
    tools: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=lambda: ["python"])
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TaskState:
    """State for a running task."""
    task_id: str
    original_request: str
    current_step: int = 0
    status: str = "running"
    tools_created: List[str] = field(default_factory=list)
    agents_created: List[str] = field(default_factory=list)
    agents_spawned: List[str] = field(default_factory=list)


class MetaAgent:
    """
    Self-extending meta-agent with sub-agent creation.
    
    Can:
    - Create and use tools (Python)
    - Create specialized sub-agents
    - Delegate tasks to sub-agents
    - Execute code in multiple languages (Python, Rust, Bash)
    - Persist everything for future reuse
    """
    
    SYSTEM_PROMPT = """You are a Meta-Agent capable of creating tools, spawning sub-agents, and executing code in multiple languages.

## Available Actions:

1. **use_tool** - Use an existing tool
   ACTION_INPUT: {{"tool_name": "...", "parameters": {{...}}}}

2. **create_tool** - Create a new Python tool
   ACTION_INPUT: {{"name": "...", "description": "...", "code": "...", "parameters": {{...}}}}

3. **execute_code** - Execute code directly
   ACTION_INPUT: {{"code": "...", "language": "python|rust|bash", "permission": "restricted|elevated"}}

4. **search_tools** - Search for existing tools
   ACTION_INPUT: {{"query": "..."}}

5. **create_agent** - Create a specialized sub-agent
   ACTION_INPUT: {{
       "name": "...",
       "description": "...",
       "capabilities": ["..."],
       "permission": "restricted|elevated",
       "languages": ["python", "rust"],
       "system_prompt": "..."
   }}

6. **spawn_agent** - Delegate a task to a sub-agent
   ACTION_INPUT: {{"agent_name": "...", "task": "..."}}

7. **search_agents** - Search for existing agents
   ACTION_INPUT: {{"query": "..."}}

8. **final_answer** - Return the final answer
   ACTION_INPUT: {{"answer": "..."}}

## Available Tools:
{tools_description}

## Available Sub-Agents:
{agents_description}

## Existing Tool Library:
{tool_library}

## Response Format:
THOUGHT: [Your reasoning]

ACTION: [action_name]

ACTION_INPUT:
```json
{{...}}
```

## Guidelines:
1. SEARCH for existing tools/agents before creating new ones
2. Create sub-agents for specialized capabilities (e.g., RustCompiler, DataAnalyzer)
3. Use ELEVATED permission only when needed (subprocess, system calls, Rust)
4. Delegate complex language-specific tasks to specialized agents

Begin!
"""

    def __init__(
        self,
        llm: LLMProvider,
        config: AgentConfig = None
    ):
        self.llm = llm
        self.config = config or AgentConfig()
        
        os.makedirs(self.config.storage_path, exist_ok=True)
        
        self._init_components()
        
        self._log(f"Meta-Agent v2.1 initialized")
        self._log(f"Model: {llm.model_name}")
        self._log(f"Docker: {self.executor.docker_available}")
    
    def _init_components(self):
        """Initialize all components."""
        # Parser
        self.parser = ResponseParser()
        
        # Tool registry
        self.tools = ToolRegistry()
        
        # Unified executor (Docker + Local fallback)
        exec_config = ExecutorConfig(
            timeout_seconds=self.config.timeout_seconds,
            memory_limit=self.config.memory_limit,
        )
        self.executor = create_executor(exec_config)
        
        # Vector store
        self.vector_store = create_vector_store(
            db_path=f"{self.config.storage_path}/memory.db"
        )
        
        # Script manager
        self.script_manager = ScriptManager(
            base_path=f"{self.config.storage_path}/scripts",
            vector_store=self.vector_store
        )
        
        # Sub-agents registry
        self.agents: Dict[str, SubAgent] = {}
        
        # Load persisted tools and agents
        self._load_persisted_tools()
        self._load_persisted_agents()
        
        # Task state
        self.current_task: Optional[TaskState] = None
    
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
    
    def _load_persisted_agents(self):
        """Load persisted agents from vector store."""
        try:
            # Search for all agents
            results = self.vector_store.search_agents("", limit=100, score_threshold=0.0)
            for r in results:
                agent = SubAgent(
                    name=r.get("name", ""),
                    description=r.get("description", ""),
                    system_prompt=r.get("system_prompt", ""),
                    capabilities=r.get("capabilities", []),
                    permission=PermissionLevel(r.get("permission", "restricted")),
                    tools=r.get("tools_used", []),
                )
                self.agents[agent.name] = agent
            
            if self.agents:
                self._log(f"Loaded {len(self.agents)} persisted agents")
        except Exception as e:
            self._log(f"Note: Could not load persisted agents: {e}")
    
    def _log(self, message: str):
        if self.config.verbose:
            print(f"[Agent] {message}")
    
    def _get_tools_description(self) -> str:
        return self.tools.get_tools_description()
    
    def _get_agents_description(self) -> str:
        if not self.agents:
            return "No sub-agents created yet."
        
        desc = []
        for name, agent in self.agents.items():
            caps = ", ".join(agent.capabilities[:3])
            langs = ", ".join(agent.languages)
            desc.append(f"• {name} [{agent.permission.value}]: {agent.description[:50]}...")
            desc.append(f"  Languages: {langs} | Capabilities: {caps}")
        return "\n".join(desc)
    
    def _get_tool_library(self) -> str:
        tools = self.script_manager.list_tools()
        if not tools:
            return "Empty - create tools to build your library."
        
        summaries = []
        for name in tools[:10]:
            info = self.script_manager.get_tool_info(name)
            if info:
                summaries.append(f"• {name}: {info.get('description', '')[:40]}...")
        return "\n".join(summaries)
    
    def _build_prompt(self, user_input: str, context: str = "") -> str:
        system = self.SYSTEM_PROMPT.format(
            tools_description=self._get_tools_description(),
            agents_description=self._get_agents_description(),
            tool_library=self._get_tool_library()
        )
        
        prompt = f"{system}\n\n"
        if context:
            prompt += f"Previous Context:\n{context}\n\n"
        prompt += f"User Request: {user_input}\n\nTHOUGHT:"
        
        return prompt
    
    # ========================================================================
    # Action Handlers
    # ========================================================================
    
    def _execute_action(self, action: str, action_input: Dict) -> str:
        handlers = {
            "use_tool": self._action_use_tool,
            "create_tool": self._action_create_tool,
            "execute_code": self._action_execute_code,
            "search_tools": self._action_search_tools,
            "create_agent": self._action_create_agent,
            "spawn_agent": self._action_spawn_agent,
            "search_agents": self._action_search_agents,
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
        tool_params = params.get("parameters", params.get("params", {}))
        
        if not tool_name:
            return "Error: No tool_name specified"
        
        return self.tools.execute(tool_name, **tool_params)
    
    def _action_create_tool(self, params: Dict) -> str:
        name = params.get("name", "")
        description = params.get("description", "")
        code = params.get("code", "")
        tool_params = params.get("parameters", {})
        
        if not name or not code:
            return "Error: Tool requires 'name' and 'code'"
        
        # Test code first
        self._log(f"Testing tool: {name}")
        result = self.executor.execute(code, language="python")
        
        if not result.success:
            return f"Tool test failed: {result.error}"
        
        # Create in registry
        tool = self.tools.create_tool_from_code(name, description, code, tool_params)
        if not tool:
            return f"Failed to create tool '{name}'"
        
        # Persist
        self.script_manager.save_tool(name, description, code, tool_params)
        
        if self.current_task:
            self.current_task.tools_created.append(name)
        
        return f"Created and persisted tool '{name}'"
    
    def _action_execute_code(self, params: Dict) -> str:
        code = params.get("code", "")
        language = params.get("language", "python").lower()
        permission_str = params.get("permission", "restricted").lower()
        
        if not code:
            return "Error: No code provided"
        
        # Parse permission
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
        
        # Execute
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
    
    def _action_create_agent(self, params: Dict) -> str:
        name = params.get("name", "")
        description = params.get("description", "")
        capabilities = params.get("capabilities", [])
        permission_str = params.get("permission", "restricted")
        languages = params.get("languages", ["python"])
        system_prompt = params.get("system_prompt", "")
        
        if not name or not description:
            return "Error: Agent requires 'name' and 'description'"
        
        # Parse permission
        try:
            permission = PermissionLevel(permission_str)
        except ValueError:
            permission = PermissionLevel.RESTRICTED
        
        # Create default system prompt if not provided
        if not system_prompt:
            system_prompt = f"""You are {name}, a specialized agent.
Description: {description}
Capabilities: {', '.join(capabilities)}
Languages: {', '.join(languages)}

You have {permission.value} permissions."""
        
        # Create agent
        agent = SubAgent(
            name=name,
            description=description,
            system_prompt=system_prompt,
            capabilities=capabilities,
            permission=permission,
            languages=languages,
        )
        
        self.agents[name] = agent
        
        # Persist to vector store
        try:
            agent_record = AgentRecord(
                name=name,
                description=description,
                system_prompt=system_prompt,
                capabilities=capabilities,
                tools_used=[],
            )
            self.vector_store.register_agent(agent_record)
        except Exception as e:
            self._log(f"Warning: Could not persist agent: {e}")
        
        if self.current_task:
            self.current_task.agents_created.append(name)
        
        return f"Created agent '{name}' with {permission.value} permissions and languages: {languages}"
    
    def _action_spawn_agent(self, params: Dict) -> str:
        agent_name = params.get("agent_name", params.get("name", ""))
        task = params.get("task", "")
        
        if not agent_name or not task:
            return "Error: spawn_agent requires 'agent_name' and 'task'"
        
        agent = self.agents.get(agent_name)
        if not agent:
            return f"Error: Agent '{agent_name}' not found. Available: {list(self.agents.keys())}"
        
        self._log(f"Spawning agent: {agent_name}")
        
        if self.current_task:
            self.current_task.agents_spawned.append(agent_name)
        
        # Build agent-specific prompt
        agent_prompt = f"""{agent.system_prompt}

Available languages: {', '.join(agent.languages)}
Permission level: {agent.permission.value}

Task: {task}

Execute this task. Use execute_code with appropriate language and permission level.

THOUGHT:"""
        
        # Run agent (simplified - uses same LLM)
        response = self.llm.generate(agent_prompt, max_tokens=2048)
        
        # Parse and execute agent's response
        parsed = self.parser.parse(response)
        
        if parsed.action:
            # Execute the agent's action with its permission level
            if parsed.action == "execute_code":
                # Override permission with agent's permission
                parsed.action_input["permission"] = agent.permission.value
            
            result = self._execute_action(parsed.action, parsed.action_input)
            return f"[{agent_name}] {result}"
        else:
            return f"[{agent_name}] {parsed.thought or response[:200]}"
    
    def _action_search_agents(self, params: Dict) -> str:
        query = params.get("query", "")
        
        if not query:
            # List all agents
            if not self.agents:
                return "No agents created yet."
            
            output = ["Available agents:"]
            for name, agent in self.agents.items():
                output.append(f"• {name} [{agent.permission.value}]: {agent.description[:50]}...")
            return "\n".join(output)
        
        # Search by capability
        results = []
        query_lower = query.lower()
        
        for name, agent in self.agents.items():
            if query_lower in name.lower() or query_lower in agent.description.lower():
                results.append(agent)
            elif any(query_lower in cap.lower() for cap in agent.capabilities):
                results.append(agent)
        
        if not results:
            return f"No agents found for: '{query}'"
        
        output = [f"Found {len(results)} agents:"]
        for agent in results:
            output.append(f"• {agent.name}: {agent.description[:50]}...")
        return "\n".join(output)
    
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
        
        self.current_task = TaskState(
            task_id=str(uuid.uuid4())[:8],
            original_request=user_input
        )
        
        full_prompt = self._build_prompt(user_input)
        final_answer = None
        
        for iteration in range(self.config.max_iterations):
            self.current_task.current_step = iteration + 1
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
        
        self.current_task.status = "completed"
        
        self._log(f"\n{'='*60}")
        self._log(f"Completed | Tools: {self.current_task.tools_created} | Agents: {self.current_task.agents_created}")
        self._log(f"{'='*60}")
        
        return final_answer


# ============================================================================
# Factory
# ============================================================================

def create_agent(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    provider: str = "huggingface",
    storage_path: str = "./agent_data",
    verbose: bool = True,
    **llm_kwargs
) -> MetaAgent:
    """Create a meta-agent."""
    
    llm = create_llm(model_name, provider, **llm_kwargs)
    
    config = AgentConfig(
        storage_path=storage_path,
        verbose=verbose
    )
    
    return MetaAgent(llm, config)


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Meta-Agent v2.1")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--provider", choices=["huggingface", "local", "vllm"], default="huggingface")
    parser.add_argument("--storage", default="./agent_data")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    Meta-Agent v2.1                            ║
║                                                               ║
║  • Create & spawn sub-agents                                 ║
║  • Multi-language: Python, Rust, Bash                        ║
║  • Permission levels: RESTRICTED, ELEVATED, SYSTEM           ║
║  • Docker isolation (no AST validation needed)               ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    llm_kwargs = {}
    if args.provider == "vllm":
        llm_kwargs["base_url"] = args.vllm_url
    
    agent = create_agent(
        model_name=args.model,
        provider=args.provider,
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
