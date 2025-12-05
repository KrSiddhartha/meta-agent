# Meta-Agent v2.2 (Latent Collaboration Edition)

A self-extending AI agent framework with advanced capabilities including sub-agent creation, multi-language code execution, persistent memory, and latent KV-cache collaboration.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Core Capabilities](#core-capabilities)
7. [Latent Collaboration](#latent-collaboration)
8. [Configuration](#configuration)
9. [API Reference](#api-reference)
10. [Testing](#testing)
11. [Examples](#examples)

---

## Overview

Meta-Agent v2.2 is a self-extending AI agent that can:
- **Create and persist tools** for reuse across sessions
- **Spawn specialized sub-agents** for complex tasks
- **Execute code** in Python, Rust, and Bash
- **Use latent collaboration** for 70-84% token savings and 4x speedup

### Version History

| Version | Features |
|---------|----------|
| v2.0 | Basic tool creation and execution |
| v2.1 | Sub-agent creation, multi-language support, Docker isolation |
| v2.2 | Latent collaboration via KV-cache transfer (LatentMAS) |

---

## Key Features

### 1. Self-Extending Tool Creation
The agent can create Python tools at runtime that persist across sessions:

```python
# Agent creates a tool automatically when needed
agent.run("Create a tool that calculates compound interest")
# Tool is saved and available in future sessions
```

### 2. Sub-Agent Spawning
Create specialized agents for specific tasks:

```python
# Create a Rust specialist agent
agent.run("Create a RustExpert agent that can compile and run Rust code")

# Later, delegate tasks to it
agent.run("Use RustExpert to implement a fast fibonacci function")
```

### 3. Multi-Language Execution
Supports Python, Rust, and Bash with appropriate permission levels:

| Language | Permission Level | Isolation |
|----------|-----------------|-----------|
| Python | RESTRICTED | Local (AST-validated) or Docker |
| Python | ELEVATED | Docker only |
| Rust | ELEVATED | Docker only |
| Bash | ELEVATED | Docker only |

### 4. Latent Collaboration (v2.2)
Based on the LatentMAS paper (arXiv:2511.20639):
- Agents communicate via KV-cache instead of text
- Hidden states encode 235-471x more information than tokens
- Only final agent decodes to text
- **70-84% fewer tokens, 4x faster**

### 5. Persistent Memory
- **VectorDB**: Semantic search over tools, agents, and context
- **ScriptManager**: Persistent tool storage with auto-loading
- **Context Memory**: Remember conversation history

---

## Architecture

```
meta_agent_v2/
├── agent.py              # Main Meta-Agent v2.1
├── agent_latent.py       # Meta-Agent v2.2 with latent collaboration
├── core/
│   ├── llm.py            # LLM provider abstraction (HuggingFace, vLLM, local)
│   ├── parser.py         # ReAct format parser with JSON repair
│   ├── tools.py          # Tool registry and execution
│   └── latent.py         # Latent collaboration module (LatentMAS)
├── sandbox/
│   ├── executor.py       # Unified code executor (Docker/Local)
│   ├── docker_sandbox.py # Docker isolation layer
│   └── ast_validator.py  # AST security validation for local execution
├── memory/
│   ├── vectordb.py       # Milvus-lite vector store
│   └── script_manager.py # Persistent tool storage
└── tests/
    ├── test_all.py       # Comprehensive test suite
    └── test_verification.py # Component verification
```

---

## Installation

### Basic Installation (Python-only, local execution)

```bash
pip install pyyaml requests
```

### Full Installation (Docker, Vector DB, Latent)

```bash
# Core dependencies
pip install pyyaml requests

# Docker support
pip install docker

# Vector database (optional, for semantic search)
pip install pymilvus sentence-transformers

# Latent collaboration (optional, requires GPU)
pip install torch transformers
```

### Docker Setup (Recommended for multi-language)

```bash
# Install Docker
# https://docs.docker.com/get-docker/

# Pull required images
docker pull python:3.12-slim
docker pull rust:1.75-slim
```

---

## Quick Start

### Basic Usage

```python
from agent import create_agent

# Create agent with HuggingFace model
agent = create_agent(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    provider="huggingface"
)

# Run a task
result = agent.run("Calculate the factorial of 10")
print(result)
```

### With Latent Collaboration

```python
from agent_latent import create_latent_meta_agent, CollaborationMode

agent = create_latent_meta_agent(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    collaboration_mode=CollaborationMode.HYBRID
)

# Complex task with latent reasoning
result = agent.run("Implement a fast prime sieve in Rust with error handling")
```

### CLI Usage

```bash
# Basic agent
python -m meta_agent_v2.agent --model mistralai/Mistral-7B-Instruct-v0.2

# With vLLM backend
python -m meta_agent_v2.agent --provider vllm --vllm-url http://localhost:8000/v1

# Latent collaboration mode
python -m meta_agent_v2.agent_latent --mode hybrid
```

---

## Core Capabilities

### Available Actions

The agent can perform these actions in its ReAct loop:

| Action | Description |
|--------|-------------|
| `use_tool` | Execute an existing tool |
| `create_tool` | Create a new Python tool |
| `execute_code` | Execute code directly (Python/Rust/Bash) |
| `search_tools` | Search for tools by capability |
| `create_agent` | Create a specialized sub-agent |
| `spawn_agent` | Delegate a task to a sub-agent |
| `search_agents` | Find agents by capability |
| `latent_reason` | Use latent pipeline for complex reasoning (v2.2) |
| `final_answer` | Return the final result |

### Tool Creation

Tools are Python functions that persist across sessions:

```python
# The agent can create tools like this:
agent.run("""
Create a tool called 'calculate_mortgage' that takes:
- principal: loan amount
- annual_rate: interest rate percentage
- years: loan term

It should return the monthly payment amount.
""")

# Tool is saved and can be used later
agent.run("Use calculate_mortgage with principal=300000, rate=6.5, years=30")
```

### Sub-Agent Creation

Create specialized agents for specific domains:

```python
agent.run("""
Create an agent called 'DataAnalyst' with these capabilities:
- Python and Pandas expertise
- Statistical analysis
- Data visualization
- Permission: ELEVATED (for matplotlib)
""")

# Delegate data tasks
agent.run("Use DataAnalyst to analyze the sales data and create a trend chart")
```

---

## Latent Collaboration

### How It Works

Traditional multi-agent systems waste tokens on text serialization between agents. Latent collaboration transfers KV-caches directly:

```
Traditional:
Agent A → [decode to text] → Agent B → [decode to text] → Agent C
(High token count, information loss)

Latent:
Agent A → [KV cache] → Agent B → [KV cache] → Agent C → [decode only once]
(70-84% fewer tokens, 4x faster)
```

### Language-Aware Configuration

Different programming languages benefit from different numbers of latent reasoning steps:

| Language | Latent Steps | Reason |
|----------|-------------|--------|
| Python | 60 | Dynamic typing requires more exploration |
| Rust | 35 | Strong type system constrains solution space |
| JavaScript | 50 | Dynamic but more constrained than Python |
| Go | 40 | Simple, explicit, low complexity |
| Bash | 30 | Short scripts need less reasoning |

### Collaboration Modes

```python
from agent_latent import CollaborationMode

# TEXT: Traditional text-based (verbose but compatible)
agent = create_latent_meta_agent(collaboration_mode=CollaborationMode.TEXT)

# LATENT: Pure KV-cache transfer (requires same model)
agent = create_latent_meta_agent(collaboration_mode=CollaborationMode.LATENT)

# HYBRID: Latent internally, text output externally (recommended)
agent = create_latent_meta_agent(collaboration_mode=CollaborationMode.HYBRID)
```

### Pipeline Configurations

#### Sequential Pipeline (Default)
```
Planner → Critic → Refiner → Solver
```

Each agent receives the previous agent's KV cache and builds upon it.

#### Hierarchical Pipeline
```
Python Specialist \
Rust Specialist    → Summarizer → Final Output
Data Specialist   /
```

Specialists work in parallel (conceptually), summarizer aggregates.

---

## Configuration

### Agent Configuration

```python
from agent import AgentConfig, PermissionLevel

config = AgentConfig(
    storage_path="./my_agent_data",  # Where to store tools/memory
    max_iterations=15,                # Max ReAct steps
    verbose=True,                     # Print debug info
    default_permission=PermissionLevel.RESTRICTED,
    timeout_seconds=60,               # Execution timeout
    memory_limit="512m"               # Docker memory limit
)
```

### Executor Configuration

```python
from sandbox.executor import ExecutorConfig

exec_config = ExecutorConfig(
    python_image="python:3.12-slim",
    rust_image="rust:1.75-slim",
    memory_limit="512m",
    cpu_limit=1.0,
    timeout_seconds=60,
    network_enabled=False,  # Disable network in sandbox
)
```

### LLM Providers

```python
from core.llm import create_llm

# HuggingFace Inference API
llm = create_llm("mistralai/Mistral-7B-Instruct-v0.2", provider="huggingface")

# vLLM server
llm = create_llm("Qwen/Qwen2.5-Coder-7B-Instruct", provider="vllm", 
                 base_url="http://localhost:8000/v1")

# Local transformers
llm = create_llm("microsoft/phi-2", provider="local")
```

---

## API Reference

### MetaAgent (v2.1)

```python
class MetaAgent:
    def __init__(self, llm: LLMProvider, config: AgentConfig = None)
    def run(self, user_input: str) -> str
```

### LatentMetaAgent (v2.2)

```python
class LatentMetaAgent:
    def __init__(self, llm, model=None, tokenizer=None, config=None)
    def run(self, user_input: str) -> str
    def run_latent(self, task: str, language: str = "python") -> str
```

### ToolRegistry

```python
class ToolRegistry:
    def register(self, tool: Tool) -> None
    def exists(self, name: str) -> bool
    def execute(self, name: str, **kwargs) -> str
    def describe_all(self) -> str
```

### UnifiedExecutor

```python
class UnifiedExecutor:
    def execute(self, code: str, language: str, 
                permission: PermissionLevel, **kwargs) -> ExecutionResult
```

### ScriptManager

```python
class ScriptManager:
    def save_tool(self, name, description, source_code, parameters=None) -> str
    def load_tool(self, name: str) -> Callable
    def list_tools(self) -> List[str]
    def search_tools(self, query: str, limit: int = 5) -> List[dict]
```

---

## Testing

### Run All Tests

```bash
cd meta_agent_v2
python -m tests.test_all
```

### Run Verification Tests

```bash
python -m tests.test_verification
```

### Expected Output

```
============================================================
        META-AGENT V2.1 - TEST SUITE
============================================================

🔒 Testing AST Security Validator...
  ✅ Safe code passes validation
  ✅ Blocks 'os' import
  ✅ Blocks 'subprocess' import
  ...

============================================================
                 TEST RESULTS
============================================================

Passed: 35/35 | Failed: 0

============================================================
✅ CERTIFICATION: ALL TESTS PASSED
============================================================
```

---

## Examples

### Example 1: Financial Calculator

```python
agent.run("""
I need to analyze an investment. Create the necessary tools and calculate:
1. Compound interest for $10,000 at 7% for 20 years
2. Monthly payment for a $500,000 mortgage at 6% for 30 years
3. Compare the total interest paid in each scenario
""")
```

### Example 2: Rust Performance Task

```python
agent.run("""
Create a RustExpert agent and have it implement a Sieve of Eratosthenes
that can find all primes up to 10 million. Benchmark it.
""")
```

### Example 3: Data Analysis

```python
agent.run("""
Analyze this CSV data and create a visualization:
- Load sales_data.csv
- Group by month and product category
- Create a stacked bar chart
- Calculate year-over-year growth
""")
```

### Example 4: Latent Reasoning

```python
from agent_latent import create_latent_meta_agent

agent = create_latent_meta_agent(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct"
)

# Complex algorithmic task
result = agent.run("""
Design and implement a concurrent web scraper in Rust that:
1. Respects robots.txt
2. Implements rate limiting
3. Handles errors gracefully
4. Stores results in SQLite
""")
```

---

## Performance

### Token Efficiency (Latent vs Text)

| Task Type | Text Mode | Latent Mode | Savings |
|-----------|-----------|-------------|---------|
| Simple | 500 tokens | 150 tokens | 70% |
| Medium | 2000 tokens | 400 tokens | 80% |
| Complex | 5000 tokens | 800 tokens | 84% |

### Execution Speed

| Component | With Docker | Without Docker |
|-----------|-------------|----------------|
| Python (RESTRICTED) | ~200ms | ~50ms |
| Python (ELEVATED) | ~300ms | ❌ Not supported |
| Rust | ~3s (compile+run) | ❌ Not supported |
| Bash | ~150ms | ❌ Not supported |

---

## Security Model

### Permission Levels

| Level | Capabilities | Isolation |
|-------|--------------|-----------|
| RESTRICTED | Pure Python, safe imports only | Local (AST-validated) |
| ELEVATED | subprocess, system calls, Rust, Bash | Docker container |
| SYSTEM | Full host access | Docker with host mounts |

### AST Validation (Local Mode)

When Docker is unavailable, Python code is validated before execution:

**Blocked:**
- `os`, `subprocess`, `sys`, `socket` imports
- `exec()`, `eval()`, `compile()` calls
- File operations with absolute paths
- Network operations

**Allowed:**
- `math`, `json`, `re`, `datetime`, `collections`
- Pure computational operations
- String manipulation

---

## Troubleshooting

### Common Issues

1. **Docker not available**
   - Install Docker: https://docs.docker.com/get-docker/
   - Agent falls back to RESTRICTED local execution

2. **Torch not installed (for latent mode)**
   - `pip install torch`
   - Agent falls back to text-only collaboration

3. **VectorDB not working**
   - `pip install pymilvus sentence-transformers`
   - Uses simple keyword search fallback

4. **HuggingFace rate limits**
   - Set `HF_TOKEN` environment variable
   - Use vLLM with local model instead

---

## License

MIT License - See LICENSE file for details.

---

## References

- [LatentMAS Paper](https://arxiv.org/abs/2511.20639) - Latent collaboration via KV-cache transfer
- [ReAct: Reasoning + Acting](https://arxiv.org/abs/2210.03629) - Foundation for agent loop
- [Milvus Lite](https://milvus.io/docs/milvus_lite.md) - Vector database

---

## Contributing

Contributions welcome! Please:
1. Run the test suite before submitting
2. Follow the existing code style
3. Add tests for new features
4. Update documentation as needed
