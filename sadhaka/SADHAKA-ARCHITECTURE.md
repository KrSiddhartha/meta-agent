# Sadhaka Architecture v0.2.0

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SADHAKA AGENT                                      │
│                     core/agent.py — ReAct Loop                                  │
│           max_iterations=10 │ run() → _react_loop() → _execute_action()         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
        ▼                               ▼                               ▼
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  PILLAR 1         │     │  PILLAR 2         │     │  PILLAR 3         │
│  Durable Memory   │     │  Explicit Tools   │     │  Specific Goals   │
│                   │     │                   │     │                   │
│  • MilvusClient   │     │  • ToolRegistry   │     │  • GoalTree       │
│  • HealthManager  │     │  • SafetyGuard    │     │  • Decomposition  │
│  • Blackboard     │     │  • AST Validation │     │  • Progress Track │
└───────────────────┘     └───────────────────┘     └───────────────────┘
        │                               │                               │
        └───────────────────────────────┼───────────────────────────────┘
                                        │
                                        ▼
                          ┌───────────────────────┐
                          │  PILLAR 4             │
                          │  Recovery Logic       │
                          │                       │
                          │  • RecoveryEngine     │
                          │  • CircuitBreaker     │
                          │  • Checkpointing      │
                          └───────────────────────┘
```

## The 4 Pillars

### PILLAR 1: Durable Memory
> "Structured, queryable memory separates agents that learn from those that loop"

```
┌─────────────────────────────────────────────────────────────────────┐
│  memory/                                                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐    ┌─────────────────────┐                │
│  │  MilvusClient       │    │  HealthManager      │                │
│  │  milvus_client.py   │    │  health_manager.py  │                │
│  │                     │    │                     │                │
│  │  • Vector search    │    │  • Health score 0-1 │                │
│  │  • CRUD operations  │    │  • Auto-deprecation │                │
│  │  • In-memory fallbk │    │  • Status tracking  │                │
│  └─────────────────────┘    └─────────────────────┘                │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Blackboard Pattern                                          │   │
│  │  orchestration/blackboard.py                                 │   │
│  │                                                              │   │
│  │  Agents don't talk directly → Read/Write shared state        │   │
│  │                                                              │   │
│  │  write_sync(key, value, agent_id)  │  watch(key, callback)   │   │
│  │  read_sync(key) → value            │  wait_for(key, timeout) │   │
│  │                                                              │   │
│  │  All state persisted to Milvus task_state collection         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### PILLAR 2: Explicit Tools
> "Clear signatures eliminate guesswork"

```
┌─────────────────────────────────────────────────────────────────────┐
│  core/tools.py                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  ToolRegistry                                                │   │
│  │                                                              │   │
│  │  Built-in: calculator │ json_parser │ string_ops             │   │
│  │                                                              │   │
│  │  create_tool(name, code) → AST validation → register         │   │
│  │  execute(name, **kwargs) → Pydantic validation → result      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────┐    ┌─────────────────────┐                │
│  │  Pydantic Schemas   │    │  SafetyGuard        │                │
│  │                     │    │  core/safety.py     │                │
│  │  input_schema: Dict │    │                     │                │
│  │  output_schema: Dict│    │  • Loop detection   │                │
│  │  JSON Schema format │    │  • Max total calls  │                │
│  │                     │    │  • Dangerous pattern│                │
│  │                     │    │  • Resource cleanup │                │
│  └─────────────────────┘    └─────────────────────┘                │
│                                                                     │
│  Dangerous Patterns Blocked:                                        │
│  rm -rf │ sudo │ eval( │ exec( │ __import__ │ os.system            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### PILLAR 3: Specific Goals
> "Precise goals anchor reasoning"

```
┌─────────────────────────────────────────────────────────────────────┐
│  goals/goal_tree.py                                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                        ┌──────────────┐                             │
│                        │  Root Goal   │                             │
│                        │  progress: ? │                             │
│                        └──────┬───────┘                             │
│                               │                                     │
│              ┌────────────────┼────────────────┐                    │
│              │                │                │                    │
│              ▼                ▼                ▼                    │
│        ┌──────────┐    ┌──────────┐    ┌──────────┐                │
│        │ Sub Goal │    │ Sub Goal │    │ Sub Goal │                │
│        │ prog: 1.0│    │ prog: 0.5│    │ prog: 0.0│                │
│        │    ✓     │    │    ◐     │    │    ○     │                │
│        └──────────┘    └──────────┘    └──────────┘                │
│                                                                     │
│  Goal Object:                                                       │
│  ├─ id, description                                                 │
│  ├─ success_criteria: List[str]  ← MEASURABLE                       │
│  ├─ input_spec, output_spec      ← EXPLICIT                         │
│  ├─ parent_id, children_ids      ← HIERARCHICAL                     │
│  ├─ status: PENDING|ACTIVE|COMPLETED|FAILED|BLOCKED                 │
│  └─ progress: 0.0 → 1.0          ← BUBBLES UP                       │
│                                                                     │
│  Methods:                                                           │
│  • create_root_goal(task_id, description, success_criteria)         │
│  • decompose_goal(parent_id, sub_goals)                             │
│  • check_alignment(goal_id, current_action) → score 0-1             │
│  • update_progress(goal_id, progress) → propagates to parent        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### PILLAR 4: Recovery Logic
> "Resilience prevents collapse"

```
┌─────────────────────────────────────────────────────────────────────┐
│  recovery/recovery_engine.py                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  RecoveryEngine                                              │   │
│  │                                                              │   │
│  │  execute_with_recovery(func, retries, backoff, fallback)     │   │
│  │                                                              │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │   │
│  │  │   Try 1     │───▶│   Try 2     │───▶│   Try 3     │      │   │
│  │  │  backoff=2s │    │  backoff=4s │    │  backoff=8s │      │   │
│  │  └─────────────┘    └─────────────┘    └──────┬──────┘      │   │
│  │                                               │              │   │
│  │                                               ▼              │   │
│  │                                        ┌─────────────┐       │   │
│  │                                        │  Fallback?  │       │   │
│  │                                        └─────────────┘       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐    │
│  │  CircuitBreaker     │    │  FailureTypes                   │    │
│  │                     │    │                                 │    │
│  │  ● CLOSED (normal)  │    │  TIMEOUT      │ backoff: 2-30s  │    │
│  │  ● HALF_OPEN (test) │    │  TOOL_ERROR   │ backoff: 1-10s  │    │
│  │  ● OPEN (blocked)   │    │  LLM_ERROR    │ backoff: 5-60s  │    │
│  │                     │    │  DOCKER_ERROR │ backoff: 3-20s  │    │
│  │  5 failures → OPEN  │    │  MCP_ERROR    │ backoff: 2-15s  │    │
│  │  30s → HALF_OPEN    │    │  UNKNOWN      │ backoff: 1-10s  │    │
│  └─────────────────────┘    └─────────────────────────────────┘    │
│                                                                     │
│  + Checkpointing: save_checkpoint(task_id, state)                   │
│  + Diagnostics: saved to Milvus diagnostics collection              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## LLM Backend: vLLM with KV-Cache Sharing

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  llm/vllm_provider.py                                                           │
│  VLLMLatentProvider — Prefix Caching for Latent Collaboration                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  SharedContext                                                            │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  System Prompt + Task Description + Blackboard State + Tools List  │  │  │
│  │  │                                                                    │  │  │
│  │  │  ══════════════════════════════════════════════════════════════   │  │  │
│  │  │           KV-Cache (computed ONCE, reused by all agents)           │  │  │
│  │  │  ══════════════════════════════════════════════════════════════   │  │  │
│  │  └────────────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                              │                                                  │
│                              │ Shared prefix                                    │
│              ┌───────────────┼───────────────┬───────────────┐                 │
│              │               │               │               │                 │
│              ▼               ▼               ▼               ▼                 │
│        ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│        │ Agent 1  │    │ Agent 2  │    │ Agent 3  │    │ Agent N  │           │
│        │ +query 1 │    │ +query 2 │    │ +query 3 │    │ +query N │           │
│        └──────────┘    └──────────┘    └──────────┘    └──────────┘           │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  TOKEN SAVINGS: 60-80%                                                   │   │
│  │                                                                          │   │
│  │  Without sharing: 3 agents × (2000 shared + 100 unique) = 6300 tokens    │   │
│  │  With sharing:    2000 shared + 3 × 100 unique          = 2300 tokens    │   │
│  │                                                                          │   │
│  │  Speedup: ~2.7x                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Methods:                                                                       │
│  • create_shared_context(id, content, warm_cache=True)                          │
│  • generate(prompt, agent_id, context_id) → single agent                        │
│  • generate_multi_agent(context_id, agent_prompts) → parallel, KV shared        │
│  • generate_with_predecessor(prompt, pred_output, context_id) → chain           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Runtime: Docker Manager

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  runtime/docker_manager.py                                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌────────────────────────────────────┐  ┌────────────────────────────────────┐│
│  │  EPHEMERAL                          │  │  PERSISTENT                        ││
│  │  Think → Test → Validate → Destroy  │  │  Deploy → Monitor → Auto-restart   ││
│  │                                     │  │                                    ││
│  │  ┌──────┐  ┌──────┐  ┌──────┐      │  │  ┌──────┐  ┌──────┐  ┌──────┐     ││
│  │  │Create│─▶│ Run  │─▶│Capture│─▶ ✗  │  │  │Start │─▶│Monitor│─▶│Health│     ││
│  │  └──────┘  └──────┘  └──────┘      │  │  └──────┘  └──────┘  └──┬───┘     ││
│  │                                     │  │                          │         ││
│  │  timeout=60s                        │  │                    ┌─────▼─────┐   ││
│  │  memory=512m                        │  │                    │ Restart?  │   ││
│  │  network=disabled                   │  │                    └───────────┘   ││
│  │                                     │  │                                    ││
│  │  Use: Testing tools, experiments    │  │  Use: MCP servers, deployed tools  ││
│  └────────────────────────────────────┘  └────────────────────────────────────┘│
│                                                                                 │
│  Fallback: If Docker unavailable → subprocess execution                        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## ReAct Loop Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ReAct Loop: Thought → Action → Observation → Repeat                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────────────┐  │
│  │ 1. Task  │───▶│ 2. Build │───▶│ 3. Create│───▶│ 4. ReAct Loop            │  │
│  │ Received │    │ Context  │    │ KV Cache │    │                          │  │
│  │          │    │          │    │          │    │   Thought                │  │
│  │ • Goal   │    │ • System │    │ • Warm   │    │      │                   │  │
│  │ • Board  │    │ • Task   │    │ • Share  │    │      ▼                   │  │
│  └──────────┘    │ • Tools  │    │          │    │   Action ────────┐       │  │
│                  └──────────┘    └──────────┘    │      │           │       │  │
│                                                  │      ▼           │       │  │
│                                                  │  Observation ◀───┘       │  │
│                                                  │      │                   │  │
│                                                  │      ▼                   │  │
│                                                  │  Final Answer?           │  │
│                                                  │   No → repeat            │  │
│                                                  │   Yes → done             │  │
│                                                  └──────────────────────────┘  │
│                                                             │                   │
│       ┌────────────────────────────────────────────────────┘                   │
│       │                                                                         │
│       ▼                                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │ 5. Exec  │───▶│ 6. Update│───▶│ 7. Goal  │───▶│ 8. Return│                  │
│  │ Tool     │    │ State    │    │ Progress │    │ Result   │                  │
│  │          │    │          │    │          │    │          │                  │
│  │ • Safety │    │ • Board  │    │ • Update │    │ • Done   │                  │
│  │ • Docker │    │ • Health │    │ • Bubble │    │ • Trace  │                  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘                  │
│                                                                                 │
│  Throughout: Tracer records THOUGHT, TOOL_CALL, TOOL_RESULT, LLM_RESPONSE      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Milvus Collections

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  memory/schemas.py — 6 Collections                                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                             │
│  │    tools    │  │   agents    │  │ mcp_servers │                             │
│  │             │  │             │  │             │                             │
│  │ id, name    │  │ id, name    │  │ id, name    │                             │
│  │ version     │  │ agent_type  │  │ container_id│                             │
│  │ source_code │  │ system_prompt│ │ source_code │                             │
│  │ input_schema│  │ tool_ids    │  │ tools_json  │                             │
│  │ health_score│  │ health_score│  │ health_score│                             │
│  │ embedding   │  │ embedding   │  │ runtime_stat│                             │
│  └─────────────┘  └─────────────┘  └─────────────┘                             │
│                                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                             │
│  │ task_state  │  │    goals    │  │ diagnostics │                             │
│  │ (Blackboard)│  │             │  │             │                             │
│  │             │  │ id, desc    │  │ failure_type│                             │
│  │ task_id     │  │ criteria    │  │ message     │                             │
│  │ inputs_json │  │ parent_id   │  │ stack_trace │                             │
│  │ outputs_json│  │ children    │  │ recoverable │                             │
│  │ intermediate│  │ status      │  │ agent_id    │                             │
│  │ owner_agent │  │ progress    │  │ timestamp   │                             │
│  └─────────────┘  └─────────────┘  └─────────────┘                             │
│                                                                                 │
│  All collections with embedding field use 1024-dim vectors (BGE-large)         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Tracing & Observability

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  core/tracing.py — Structured Execution Traces                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  TraceEventTypes:                                                               │
│  ┌────────────┬────────────┬────────────┬────────────┬────────────┐            │
│  │ TASK_START │  THOUGHT   │ TOOL_CALL  │TOOL_RESULT │LLM_RESPONSE│            │
│  ├────────────┼────────────┼────────────┼────────────┼────────────┤            │
│  │   ERROR    │  RECOVERY  │GOAL_UPDATE │ BLACKBOARD │  TASK_END  │            │
│  └────────────┴────────────┴────────────┴────────────┴────────────┘            │
│                                                                                 │
│  Example Timeline Output:                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  0.000s [task_start]    task_id: task_abc123                             │  │
│  │  0.001s [thought]       "I need to calculate the sum..."                 │  │
│  │  0.002s [tool_call]     calculator({"expression": "2+2"})                │  │
│  │  0.015s [tool_result]   ✓ {"success": true, "result": 4}                 │  │
│  │  0.520s [llm_response]  150 tokens, 505ms, kv_cache_hit=true             │  │
│  │  0.521s [task_end]      success=true                                     │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  Summary Metrics:                                                               │
│  • total_events, llm_calls, llm_tokens, tool_calls, errors                      │
│  • avg_iteration_time_ms, duration_seconds                                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
sadhaka-0.2.0/
├── __init__.py              # Package entry, version="0.2.0"
├── config.py                # SadhakaConfig dataclass
├── README.md                # Quick start
├── CHANGELOG.md             # Version history
├── pyproject.toml           # Modern Python packaging
│
├── core/
│   ├── agent.py             # SadhakaAgent + ReAct loop
│   ├── parser.py            # ReActParser (ReAct + JSON formats)
│   ├── tools.py             # ToolRegistry + built-ins
│   ├── tracing.py           # NEW: Tracer + ExecutionTrace
│   └── safety.py            # NEW: SafetyGuard + LoopDetector
│
├── llm/
│   └── vllm_provider.py     # VLLMLatentProvider + MockLLM
│
├── memory/
│   ├── milvus_client.py     # MilvusClient + in-memory fallback
│   ├── health_manager.py    # HealthManager + auto-deprecation
│   └── schemas.py           # All 6 collection schemas
│
├── orchestration/
│   └── blackboard.py        # Blackboard pattern
│
├── goals/
│   └── goal_tree.py         # GoalTree + hierarchical goals
│
├── recovery/
│   └── recovery_engine.py   # RecoveryEngine + CircuitBreaker
│
├── runtime/
│   └── docker_manager.py    # DockerManager + subprocess fallback
│
├── mcp/
│   └── factory.py           # MCPServerFactory (placeholder)
│
└── tests/
    └── test_all.py          # 42/42 tests passing
```

## Test Results

```
============================================================
SADHAKA TEST SUITE — v0.2.0
============================================================

[VLLMLatentProvider]    5/5 ✓
[MilvusClient]          4/4 ✓
[HealthManager]         3/3 ✓
[Blackboard]            4/4 ✓
[DockerManager]         4/4 ✓
[GoalTree]              5/5 ✓
[RecoveryEngine]        5/5 ✓
[ReActParser]           4/4 ✓
[ToolRegistry]          7/7 ✓
[Integration]           1/1 ✓

============================================================
Results: 42/42 passed
============================================================
```

## Graceful Degradation

| Component | If Unavailable | Fallback |
|-----------|----------------|----------|
| vLLM | No GPU / not installed | MockLLM (test responses) |
| Milvus | Not running | In-memory dict storage |
| Docker | Not installed | subprocess execution |

All core functionality works without optional dependencies.
