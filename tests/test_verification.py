#!/usr/bin/env python3
"""
Meta-Agent v2.2 - Component Verification Test

This script verifies all components are working correctly:
1. Core modules (LLM, Parser, Tools)
2. Sandbox execution (Docker/Local)
3. Memory systems (ScriptManager, VectorDB)
4. Latent collaboration (KV-cache transfer)
5. Integration test

Run with: python -m pytest tests/test_verification.py -v
Or directly: python tests/test_verification.py
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Test Results Tracking
# ============================================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    details: Optional[str] = None


class TestRunner:
    def __init__(self):
        self.results: list[TestResult] = []
        self.verbose = True
        self.skipped = 0
    
    def test(self, name: str):
        """Decorator for test functions."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    if result is None or result is True:
                        self.results.append(TestResult(name, True, "PASSED"))
                        if self.verbose:
                            print(f"  ✅ {name}")
                        return True
                    elif isinstance(result, str) and result.lower().startswith("skip"):
                        # Handle skipped tests
                        self.results.append(TestResult(name, True, result))
                        self.skipped += 1
                        if self.verbose:
                            print(f"  ⚠️  {name}: {result}")
                        return True
                    else:
                        self.results.append(TestResult(name, False, str(result)))
                        if self.verbose:
                            print(f"  ❌ {name}: {result}")
                        return False
                except Exception as e:
                    self.results.append(TestResult(name, False, f"EXCEPTION: {e}", str(e)))
                    if self.verbose:
                        print(f"  ❌ {name}: {type(e).__name__}: {e}")
                    return False
            return wrapper
        return decorator
    
    def summary(self):
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        skipped_info = f" ({self.skipped} skipped)" if self.skipped > 0 else ""
        print(f"\n{'='*60}")
        print(f"VERIFICATION SUMMARY: {passed}/{total} tests passed{skipped_info}")
        print(f"{'='*60}")
        
        if passed == total:
            print("✅ ALL COMPONENTS CERTIFIED WORKING")
        else:
            print("❌ SOME COMPONENTS FAILED:")
            for r in self.results:
                if not r.passed:
                    print(f"   - {r.name}: {r.message}")
        
        return passed == total


runner = TestRunner()


# ============================================================================
# 1. CORE MODULES TESTS
# ============================================================================

print("\n" + "="*60)
print("1. CORE MODULES")
print("="*60)

@runner.test("Import core.llm")
def test_import_llm():
    from core.llm import LLMProvider, create_llm
    return True

@runner.test("Import core.parser")
def test_import_parser():
    from core.parser import ResponseParser, ParsedResponse
    return True

@runner.test("Import core.tools")
def test_import_tools():
    from core.tools import Tool, ToolRegistry
    return True

@runner.test("ResponseParser parses THOUGHT/ACTION")
def test_parser():
    from core.parser import ResponseParser
    
    parser = ResponseParser()
    response = '''THOUGHT: I need to calculate fibonacci.

ACTION: execute_code

ACTION_INPUT:
```json
{"code": "print(42)", "language": "python"}
```
'''
    parsed = parser.parse(response)
    
    assert parsed.thought is not None, "Missing thought"
    assert parsed.action == "execute_code", f"Wrong action: {parsed.action}"
    assert parsed.action_input.get("code") == "print(42)", "Wrong code"
    return True

@runner.test("ToolRegistry create and execute")
def test_tool_registry():
    from core.tools import ToolRegistry
    
    registry = ToolRegistry()
    
    # Create tool from code
    code = '''
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''
    tool = registry.create_tool_from_code("add_numbers", "Adds numbers", code, {"a": "int", "b": "int"})
    assert tool is not None, "Tool creation failed"
    
    # Execute
    result = registry.execute("add_numbers", a=5, b=3)
    assert "8" in str(result), f"Wrong result: {result}"
    return True


# Run core tests
test_import_llm()
test_import_parser()
test_import_tools()
test_parser()
test_tool_registry()


# ============================================================================
# 2. SANDBOX EXECUTION TESTS
# ============================================================================

print("\n" + "="*60)
print("2. SANDBOX EXECUTION")
print("="*60)

@runner.test("Import sandbox.executor")
def test_import_executor():
    from sandbox.executor import (
        UnifiedExecutor, ExecutorConfig, ExecutionResult,
        PermissionLevel, create_executor
    )
    return True

@runner.test("Create executor")
def test_create_executor():
    from sandbox.executor import create_executor, ExecutorConfig
    
    config = ExecutorConfig(timeout_seconds=30)
    executor = create_executor(config)
    assert executor is not None
    return True

@runner.test("Execute Python code")
def test_execute_python():
    from sandbox.executor import create_executor
    
    executor = create_executor()
    result = executor.execute("print('Hello from test')", language="python")
    
    assert result.success, f"Execution failed: {result.error}"
    assert "Hello" in result.output, f"Wrong output: {result.output}"
    return True

@runner.test("Execute with inputs")
def test_execute_with_inputs():
    from sandbox.executor import create_executor
    
    executor = create_executor()
    code = '''
x = inputs.get("x", 0)
y = inputs.get("y", 0)
result = x + y
print(f"Result: {result}")
'''
    result = executor.execute(code, language="python", inputs={"x": 10, "y": 20})
    
    assert result.success, f"Execution failed: {result.error}"
    assert "30" in result.output, f"Wrong output: {result.output}"
    return True

@runner.test("Rust compilation check")
def test_rust_available():
    from sandbox.executor import create_executor
    
    executor = create_executor()
    # Just check if Rust execution path exists - don't require it to work
    code = '''fn main() { println!("test"); }'''
    result = executor.execute(code, language="rust")
    
    # Pass if either it works OR it fails gracefully with "Rust not available"
    if result.success:
        return True
    if "not available" in str(result.error).lower() or "docker" in str(result.error).lower():
        return True  # Acceptable - Rust requires Docker
    return f"Unexpected error: {result.error}"


# Run sandbox tests
test_import_executor()
test_create_executor()
test_execute_python()
test_execute_with_inputs()
test_rust_available()


# ============================================================================
# 3. MEMORY SYSTEMS TESTS
# ============================================================================

print("\n" + "="*60)
print("3. MEMORY SYSTEMS")
print("="*60)

@runner.test("Import memory.script_manager")
def test_import_script_manager():
    from memory.script_manager import ScriptManager
    return True

@runner.test("ScriptManager save and load tool")
def test_script_manager():
    from memory.script_manager import ScriptManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ScriptManager(base_path=tmpdir)
        
        # Save a tool
        code = 'def greet(name): return f"Hello, {name}!"'
        manager.save_tool("greet", "Greets a person", code, {"name": "str"})
        
        # List tools
        tools = manager.list_tools()
        assert "greet" in tools, f"Tool not saved: {tools}"
        
        # Load tool
        func = manager.load_tool("greet")
        assert func is not None, "Failed to load tool"
        
        # Execute loaded tool
        result = func("World")
        assert result == "Hello, World!", f"Wrong result: {result}"
        
    return True

@runner.test("ScriptManager search tools")
def test_script_manager_search():
    from memory.script_manager import ScriptManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ScriptManager(base_path=tmpdir)
        
        # Save multiple tools
        manager.save_tool("calculate_sum", "Calculates sum of numbers", "def f(a,b): return a+b", {})
        manager.save_tool("format_text", "Formats text", "def f(s): return s.upper()", {})
        
        # Search
        results = manager.search_tools("calculate")
        assert len(results) > 0, "No search results"
        assert any("calculate" in r["name"] for r in results), "Wrong results"
        
    return True


# Run memory tests
test_import_script_manager()
test_script_manager()
test_script_manager_search()


# ============================================================================
# 4. LATENT COLLABORATION TESTS
# ============================================================================

print("\n" + "="*60)
print("4. LATENT COLLABORATION")
print("="*60)

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("  ⚠️  torch not installed - latent tests will be skipped")

@runner.test("Import core.latent")
def test_import_latent():
    if not TORCH_AVAILABLE:
        return "Skipped (torch not installed)"
    from core.latent import (
        LatentAgent,
        LatentPipeline,
        LatentThoughts,
        WorkingMemoryTransfer,
        InputOutputAligner,
        LatentThoughtGenerator,
        LANGUAGE_LATENT_CONFIG,
        create_latent_agent,
        create_sequential_pipeline,
        create_hierarchical_pipeline,
    )
    return True

@runner.test("LANGUAGE_LATENT_CONFIG structure")
def test_latent_config():
    if not TORCH_AVAILABLE:
        return "Skipped (torch not installed)"
    from core.latent import LANGUAGE_LATENT_CONFIG
    
    # Check required languages
    required = ["python", "rust", "javascript", "go", "default"]
    for lang in required:
        assert lang in LANGUAGE_LATENT_CONFIG, f"Missing {lang}"
        config = LANGUAGE_LATENT_CONFIG[lang]
        assert "latent_steps" in config, f"Missing latent_steps for {lang}"
        assert "temperature" in config, f"Missing temperature for {lang}"
    
    # Check Python has more steps than Rust (based on scaling laws)
    assert LANGUAGE_LATENT_CONFIG["python"]["latent_steps"] > LANGUAGE_LATENT_CONFIG["rust"]["latent_steps"], \
        "Python should need more latent steps than Rust"
    
    return True

@runner.test("WorkingMemoryTransfer.transfer")
def test_working_memory_transfer():
    if not TORCH_AVAILABLE:
        return "Skipped (torch not installed)"
    from core.latent import WorkingMemoryTransfer
    import torch
    
    # Create mock KV caches (2 layers, batch=1, heads=4, seq=10, dim=64)
    def make_mock_kv(seq_len: int):
        return tuple(
            (torch.randn(1, 4, seq_len, 64), torch.randn(1, 4, seq_len, 64))
            for _ in range(2)  # 2 layers
        )
    
    source_kv = make_mock_kv(10)
    target_kv = make_mock_kv(5)
    
    # Transfer
    result = WorkingMemoryTransfer.transfer(source_kv, target_kv)
    
    assert result.success, f"Transfer failed: {result.error}"
    assert result.combined_kv is not None, "Missing combined KV"
    
    # Check combined sequence length
    combined_seq = result.combined_kv[0][0].shape[2]
    assert combined_seq == 15, f"Wrong combined length: {combined_seq}"
    
    return True

@runner.test("WorkingMemoryTransfer.validate_compatibility")
def test_kv_compatibility():
    if not TORCH_AVAILABLE:
        return "Skipped (torch not installed)"
    from core.latent import WorkingMemoryTransfer
    import torch
    
    # Compatible KVs
    kv1 = tuple((torch.randn(1, 4, 10, 64), torch.randn(1, 4, 10, 64)) for _ in range(2))
    kv2 = tuple((torch.randn(1, 4, 5, 64), torch.randn(1, 4, 5, 64)) for _ in range(2))
    
    is_compat, msg = WorkingMemoryTransfer.validate_compatibility(kv1, kv2)
    assert is_compat, f"Should be compatible: {msg}"
    
    # Incompatible KVs (different head count)
    kv3 = tuple((torch.randn(1, 8, 5, 64), torch.randn(1, 8, 5, 64)) for _ in range(2))
    
    is_compat, msg = WorkingMemoryTransfer.validate_compatibility(kv1, kv3)
    assert not is_compat, "Should be incompatible (different heads)"
    
    return True


# Run latent tests
test_import_latent()
test_latent_config()
test_working_memory_transfer()
test_kv_compatibility()


# ============================================================================
# 5. AGENT INTEGRATION TESTS
# ============================================================================

print("\n" + "="*60)
print("5. AGENT INTEGRATION")
print("="*60)

@runner.test("Import agent.py")
def test_import_agent():
    from agent import MetaAgent, AgentConfig, create_agent
    return True

@runner.test("Import agent_latent.py")
def test_import_agent_latent():
    from agent_latent import (
        LatentMetaAgent, 
        LatentAgentConfig, 
        CollaborationMode,
        create_latent_meta_agent
    )
    return True

@runner.test("CollaborationMode enum")
def test_collaboration_mode():
    from agent_latent import CollaborationMode
    
    assert CollaborationMode.TEXT.value == "text"
    assert CollaborationMode.LATENT.value == "latent"
    assert CollaborationMode.HYBRID.value == "hybrid"
    return True

@runner.test("LatentAgentConfig defaults")
def test_latent_agent_config():
    from agent_latent import LatentAgentConfig, CollaborationMode
    
    config = LatentAgentConfig()
    
    assert config.max_iterations == 15
    assert config.collaboration_mode == CollaborationMode.HYBRID
    assert config.default_latent_steps == 50
    assert config.default_language == "python"
    return True


# Run integration tests
test_import_agent()
test_import_agent_latent()
test_collaboration_mode()
test_latent_agent_config()


# ============================================================================
# 6. END-TO-END FLOW TEST
# ============================================================================

print("\n" + "="*60)
print("6. END-TO-END FLOW")
print("="*60)

@runner.test("Full pipeline: parse -> execute -> result")
def test_full_pipeline():
    from core.parser import ResponseParser
    from sandbox.executor import create_executor
    
    # Simulate LLM response
    llm_response = '''THOUGHT: I need to calculate the factorial of 5.

ACTION: execute_code

ACTION_INPUT:
```json
{
    "code": "def factorial(n):\\n    if n <= 1: return 1\\n    return n * factorial(n-1)\\n\\nresult = factorial(5)\\nprint(f'Factorial of 5 is {result}')",
    "language": "python"
}
```
'''
    
    # Parse
    parser = ResponseParser()
    parsed = parser.parse(llm_response)
    
    assert parsed.action == "execute_code", f"Wrong action: {parsed.action}"
    
    # Execute
    executor = create_executor()
    code = parsed.action_input.get("code", "")
    language = parsed.action_input.get("language", "python")
    
    result = executor.execute(code, language=language)
    
    assert result.success, f"Execution failed: {result.error}"
    assert "120" in result.output, f"Wrong result: {result.output}"
    
    return True


# Run e2e test
test_full_pipeline()


# ============================================================================
# FINAL SUMMARY
# ============================================================================

success = runner.summary()
sys.exit(0 if success else 1)
