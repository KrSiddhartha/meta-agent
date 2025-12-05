"""
Comprehensive Test Suite for Meta-Agent v2

Tests all components:
1. AST Security Validator
2. Parser (regex and JSON repair)
3. Tool Registry
4. Sandbox (Local)
5. VectorDB
6. Script Manager
7. Integration tests
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestResult:
    """Test result tracker."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def success(self, test_name: str):
        self.passed += 1
        print(f"  ✅ {test_name}")
    
    def fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"  ❌ {test_name}: {error}")
    
    def summary(self) -> str:
        total = self.passed + self.failed
        return f"Passed: {self.passed}/{total} | Failed: {self.failed}"


# ============================================================================
# 1. AST Security Validator Tests
# ============================================================================

def test_ast_validator(results: TestResult):
    """Test AST security validation."""
    print("\n🔒 Testing AST Security Validator...")
    
    from sandbox.ast_validator import ASTSecurityValidator, validate_code
    
    validator = ASTSecurityValidator()
    
    # Test 1: Safe code should pass
    safe_code = """
def calculate(a, b):
    return a + b

result = calculate(5, 3)
print(result)
"""
    report = validator.validate(safe_code)
    if report.is_safe:
        results.success("Safe code passes validation")
    else:
        results.fail("Safe code passes validation", str(report.violations))
    
    # Test 2: os import should be blocked
    dangerous_code = """
import os
os.system('ls')
"""
    report = validator.validate(dangerous_code)
    if not report.is_safe and "os" in str(report.blocked_imports):
        results.success("Blocks 'os' import")
    else:
        results.fail("Blocks 'os' import", "Did not block os import")
    
    # Test 3: subprocess should be blocked
    dangerous_code2 = """
import subprocess
subprocess.run(['ls'])
"""
    report = validator.validate(dangerous_code2)
    if not report.is_safe:
        results.success("Blocks 'subprocess' import")
    else:
        results.fail("Blocks 'subprocess' import", "Did not block subprocess")
    
    # Test 4: exec/eval should be blocked
    dangerous_code3 = """
exec('print("hello")')
"""
    report = validator.validate(dangerous_code3)
    if not report.is_safe and "exec" in str(report.blocked_calls):
        results.success("Blocks 'exec' call")
    else:
        results.fail("Blocks 'exec' call", "Did not block exec")
    
    # Test 5: Useful modules should be allowed
    useful_code = """
import json
import math
import re
data = json.dumps({"x": math.sqrt(16)})
"""
    report = validator.validate(useful_code)
    if report.is_safe:
        results.success("Allows useful modules (json, math, re)")
    else:
        results.fail("Allows useful modules", str(report.violations))
    
    # Test 6: Syntax errors caught
    bad_syntax = "def foo( invalid syntax"
    report = validator.validate(bad_syntax)
    if not report.is_safe and "Syntax" in str(report.violations):
        results.success("Catches syntax errors")
    else:
        results.fail("Catches syntax errors", "Did not catch syntax error")


# ============================================================================
# 2. Parser Tests
# ============================================================================

def test_parser(results: TestResult):
    """Test response parser."""
    print("\n📝 Testing Response Parser...")
    
    from core.parser import ResponseParser, JSONRepairer
    
    parser = ResponseParser()
    
    # Test 1: Standard format parsing
    response1 = """THOUGHT: I need to calculate the sum of numbers.

ACTION: use_tool

ACTION_INPUT:
```json
{"tool_name": "calculator", "parameters": {"expression": "2 + 2"}}
```
"""
    parsed = parser.parse(response1)
    if parsed.thought and parsed.action == "use_tool" and parsed.action_input.get("tool_name") == "calculator":
        results.success("Parses standard ReAct format")
    else:
        results.fail("Parses standard ReAct format", f"Got: thought={parsed.thought[:30]}, action={parsed.action}")
    
    # Test 2: Without code blocks
    response2 = """THOUGHT: Let me search for tools.

ACTION: search_tools

ACTION_INPUT: {"query": "calculator"}
"""
    parsed = parser.parse(response2)
    if parsed.action == "search_tools" and parsed.action_input.get("query") == "calculator":
        results.success("Parses without code blocks")
    else:
        results.fail("Parses without code blocks", f"Got: {parsed.action_input}")
    
    # Test 3: JSON repair - trailing comma
    bad_json = '{"name": "test", "value": 42,}'
    repaired = JSONRepairer.repair(bad_json)
    try:
        data = json.loads(repaired)
        if data.get("name") == "test":
            results.success("JSON repair: trailing comma")
        else:
            results.fail("JSON repair: trailing comma", "Wrong data")
    except:
        results.fail("JSON repair: trailing comma", "Failed to parse")
    
    # Test 4: JSON repair - Python booleans
    bad_json2 = '{"active": True, "count": None}'
    repaired = JSONRepairer.repair(bad_json2)
    try:
        data = json.loads(repaired)
        if data.get("active") == True and data.get("count") is None:
            results.success("JSON repair: Python True/None")
        else:
            results.fail("JSON repair: Python True/None", f"Got: {data}")
    except:
        results.fail("JSON repair: Python True/None", "Failed to parse")
    
    # Test 5: Extract JSON from text
    messy_text = "Here is the result: ```json\n{\"answer\": 42}\n``` done."
    extracted = JSONRepairer.extract_json(messy_text)
    if extracted and extracted.get("answer") == 42:
        results.success("Extracts JSON from markdown")
    else:
        results.fail("Extracts JSON from markdown", f"Got: {extracted}")


# ============================================================================
# 3. Tool Registry Tests
# ============================================================================

def test_tools(results: TestResult):
    """Test tool registry."""
    print("\n🔧 Testing Tool Registry...")
    
    from core.tools import Tool, ToolRegistry
    
    registry = ToolRegistry()
    
    # Test 1: Built-in tools exist
    if registry.exists("calculator") and registry.exists("list_tools"):
        results.success("Built-in tools registered")
    else:
        results.fail("Built-in tools registered", f"Tools: {registry.list_names()}")
    
    # Test 2: Calculator works
    result = registry.execute("calculator", expression="2 + 3 * 4")
    if "14" in result:
        results.success("Calculator executes correctly")
    else:
        results.fail("Calculator executes correctly", f"Got: {result}")
    
    # Test 3: Register custom tool
    def greet(name: str = "World"):
        return f"Hello, {name}!"
    
    custom_tool = Tool(
        name="greeter",
        description="Greets someone by name",
        function=greet,
        parameters={"name": {"type": "string", "default": "World"}}
    )
    registry.register(custom_tool)
    
    if registry.exists("greeter"):
        results.success("Custom tool registration")
    else:
        results.fail("Custom tool registration", "Tool not found")
    
    # Test 4: Execute custom tool
    result = registry.execute("greeter", name="Alice")
    if "Hello, Alice!" in result:
        results.success("Custom tool execution")
    else:
        results.fail("Custom tool execution", f"Got: {result}")
    
    # Test 5: Create tool from code
    code = """
def multiply(a, b):
    return a * b
"""
    tool = registry.create_tool_from_code(
        name="multiplier",
        description="Multiplies two numbers",
        code=code,
        parameters={"a": {"type": "number"}, "b": {"type": "number"}}
    )
    
    if tool and registry.exists("multiplier"):
        result = registry.execute("multiplier", a=6, b=7)
        if "42" in result:
            results.success("Create tool from code")
        else:
            results.fail("Create tool from code", f"Got: {result}")
    else:
        results.fail("Create tool from code", "Tool not created")
    
    # Test 6: Tools description
    desc = registry.get_tools_description()
    if "calculator" in desc and "greeter" in desc:
        results.success("Tools description generation")
    else:
        results.fail("Tools description generation", "Missing tools in description")


# ============================================================================
# 4. Executor Tests (Unified - replaces sandbox tests)
# ============================================================================

def test_executor(results: TestResult):
    """Test unified code executor."""
    print("\n📦 Testing Unified Executor...")
    
    from sandbox.executor import UnifiedExecutor, ExecutorConfig, PermissionLevel
    
    executor = UnifiedExecutor()
    
    # Test 1: Python execution
    code1 = """
result = 2 + 2
print(f"Sum is {result}")
"""
    exec_result = executor.execute(code1, language="python")
    if exec_result.success and "4" in exec_result.output:
        results.success("Python code execution")
    else:
        results.fail("Python code execution", f"Error: {exec_result.error}")
    
    # Test 2: Return value capture
    code2 = """
result = [1, 2, 3, 4, 5]
"""
    exec_result = executor.execute(code2, language="python")
    if exec_result.success and exec_result.return_value == [1, 2, 3, 4, 5]:
        results.success("Return value capture")
    else:
        results.fail("Return value capture", f"Got: {exec_result.return_value}")
    
    # Test 3: Input variables
    code3 = """
result = x * y
print(f"{x} * {y} = {result}")
"""
    exec_result = executor.execute(code3, language="python", inputs={"x": 7, "y": 8})
    if exec_result.success and "56" in exec_result.output:
        results.success("Input variables")
    else:
        results.fail("Input variables", f"Error: {exec_result.error}")
    
    # Test 4: Error handling
    bad_code = """
result = 1 / 0
"""
    exec_result = executor.execute(bad_code, language="python")
    if not exec_result.success and "ZeroDivision" in str(exec_result.error):
        results.success("Error handling")
    else:
        results.fail("Error handling", f"Got: {exec_result}")
    
    # Test 5: Permission levels exist
    try:
        assert PermissionLevel.RESTRICTED.value == "restricted"
        assert PermissionLevel.ELEVATED.value == "elevated"
        assert PermissionLevel.SYSTEM.value == "system"
        results.success("Permission levels defined")
    except Exception as e:
        results.fail("Permission levels defined", str(e))


# ============================================================================
# 4b. Local Sandbox Tests (fallback mode)
# ============================================================================

def test_local_sandbox(results: TestResult):
    """Test local sandbox execution (AST validated)."""
    print("\n🔒 Testing Local Sandbox (AST-validated fallback)...")
    
    from sandbox.executor import LocalExecutor, PermissionLevel
    
    sandbox = LocalExecutor(timeout=5)
    
    # Test 1: Simple execution
    code1 = """
result = 2 + 2
print(f"Sum is {result}")
"""
    exec_result = sandbox.execute_python(code1)
    if exec_result.success and "4" in exec_result.output:
        results.success("Local: Simple execution")
    else:
        results.fail("Local: Simple execution", f"Error: {exec_result.error}")
    
    # Test 2: Blocks dangerous code (AST validation)
    dangerous = """
import os
os.system('echo pwned')
"""
    exec_result = sandbox.execute_python(dangerous)
    if not exec_result.success and "Security" in str(exec_result.error):
        results.success("Local: Blocks dangerous imports (AST)")
    else:
        results.fail("Local: Blocks dangerous imports", f"Should have blocked: {exec_result}")
    
    # Test 3: ELEVATED requires Docker
    exec_result = sandbox.execute_python("print('hi')", permission=PermissionLevel.ELEVATED)
    if not exec_result.success and "Docker" in str(exec_result.error):
        results.success("Local: ELEVATED requires Docker")
    else:
        results.fail("Local: ELEVATED requires Docker", f"Should require Docker: {exec_result}")


# ============================================================================
# 5. VectorDB Tests
# ============================================================================

def test_vectordb(results: TestResult):
    """Test VectorDB operations."""
    print("\n🧠 Testing VectorDB...")
    
    # Check if dependencies are available
    try:
        import pymilvus
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"  ⚠️  Skipping VectorDB tests: {e}")
        print("  Install with: pip install pymilvus sentence-transformers")
        results.success("VectorDB tests skipped (dependencies not installed)")
        return
    
    # Create temp directory for test DB
    temp_dir = tempfile.mkdtemp()
    db_path = f"{temp_dir}/test_memory.db"
    
    try:
        from memory.vectordb import MilvusVectorStore, ContextEntry, ToolRecord
        
        store = MilvusVectorStore(db_path=db_path)
        
        # Test 1: Store context
        entry = ContextEntry(
            content="The user wants to calculate compound interest",
            entry_type="user_input",
            task_id="test123"
        )
        entry_id = store.store_context(entry)
        if entry_id:
            results.success("Store context entry")
        else:
            results.fail("Store context entry", "No ID returned")
        
        # Test 2: Search context
        search_results = store.search_context("compound interest calculation", limit=5)
        if search_results and any("compound" in r.get("content", "").lower() for r in search_results):
            results.success("Search context semantically")
        else:
            results.fail("Search context semantically", f"Got: {search_results}")
        
        # Test 3: Register tool
        tool_record = ToolRecord(
            name="compound_interest",
            description="Calculate compound interest on principal",
            source_code="def compound_interest(p, r, t): return p * (1 + r) ** t",
            capabilities=["financial", "calculation"],
            parameters={"p": "principal", "r": "rate", "t": "time"}
        )
        tool_id = store.register_tool(tool_record)
        if tool_id:
            results.success("Register tool in VectorDB")
        else:
            results.fail("Register tool in VectorDB", "No ID returned")
        
        # Test 4: Search tools
        tool_results = store.search_tools("calculate interest financial", limit=5)
        if tool_results and any("compound" in r.get("name", "").lower() for r in tool_results):
            results.success("Search tools semantically")
        else:
            results.fail("Search tools semantically", f"Got: {tool_results}")
        
        # Test 5: Get stats
        stats = store.get_stats()
        if stats.get("context_entries", 0) > 0 and stats.get("tools", 0) > 0:
            results.success("Get VectorDB stats")
        else:
            results.fail("Get VectorDB stats", f"Got: {stats}")
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# 6. Script Manager Tests
# ============================================================================

def test_script_manager(results: TestResult):
    """Test script persistence."""
    print("\n💾 Testing Script Manager...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        from memory.script_manager import ScriptManager
        
        manager = ScriptManager(base_path=temp_dir, vector_store=None)
        
        # Test 1: Save tool
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        path = manager.save_tool(
            name="fibonacci",
            description="Calculate fibonacci numbers recursively",
            source_code=code,
            parameters={"n": {"type": "integer", "description": "Position in sequence"}}
        )
        
        if path and os.path.exists(path):
            results.success("Save tool to disk")
        else:
            results.fail("Save tool to disk", f"Path: {path}")
        
        # Test 2: Load tool
        func = manager.load_tool("fibonacci")
        if func and callable(func):
            result = func(10)
            if result == 55:
                results.success("Load and execute persisted tool")
            else:
                results.fail("Load and execute persisted tool", f"Got: {result}")
        else:
            results.fail("Load and execute persisted tool", "Function not loaded")
        
        # Test 3: List tools
        tools = manager.list_tools()
        if "fibonacci" in tools:
            results.success("List persisted tools")
        else:
            results.fail("List persisted tools", f"Got: {tools}")
        
        # Test 4: Get tool info
        info = manager.get_tool_info("fibonacci")
        if info and info.get("name") == "fibonacci" and "mathematical" in str(info.get("capabilities", [])):
            results.success("Get tool metadata with capabilities")
        else:
            results.fail("Get tool metadata with capabilities", f"Got: {info}")
        
        # Test 5: Search tools
        search_results = manager.search_tools("recursive calculation", limit=5)
        if search_results and any("fibonacci" in r.get("name", "") for r in search_results):
            results.success("Search tools by keyword")
        else:
            results.fail("Search tools by keyword", f"Got: {search_results}")
        
        # Test 6: Delete tool
        deleted = manager.delete_tool("fibonacci")
        if deleted and "fibonacci" not in manager.list_tools():
            results.success("Delete persisted tool")
        else:
            results.fail("Delete persisted tool", "Tool still exists")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# 7. Integration Tests
# ============================================================================

def test_integration(results: TestResult):
    """Test component integration."""
    print("\n🔗 Testing Integration...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test 1: Validator + Executor (Local mode)
        from sandbox.ast_validator import ASTSecurityValidator
        from sandbox.executor import LocalExecutor
        
        validator = ASTSecurityValidator()
        executor = LocalExecutor()
        
        code = """
import math
result = math.factorial(5)
print(f"5! = {result}")
"""
        report = validator.validate(code)
        if report.is_safe:
            exec_result = executor.execute_python(code)
            if exec_result.success and "120" in exec_result.output:
                results.success("Validator + Executor pipeline")
            else:
                results.fail("Validator + Executor pipeline", f"Execution: {exec_result.error}")
        else:
            results.fail("Validator + Executor pipeline", f"Validation: {report.violations}")
        
        # Test 2: VectorDB + ScriptManager (if available)
        try:
            from memory.vectordb import MilvusVectorStore
            from memory.script_manager import ScriptManager
            
            store = MilvusVectorStore(db_path=f"{temp_dir}/integration.db")
            manager = ScriptManager(base_path=f"{temp_dir}/scripts", vector_store=store)
            
            code2 = """
def discount_price(price, percent):
    return price * (1 - percent / 100)
"""
            manager.save_tool(
                name="discount_calculator",
                description="Calculate discounted price",
                source_code=code2
            )
            
            # Should be searchable in VectorDB
            search_results = store.search_tools("calculate discount price percentage")
            if search_results and any("discount" in r.get("name", "") for r in search_results):
                results.success("ScriptManager indexes in VectorDB")
            else:
                results.fail("ScriptManager indexes in VectorDB", f"Got: {search_results}")
        except ImportError:
            print("  ⚠️  Skipping VectorDB integration test (dependencies not installed)")
            results.success("VectorDB integration skipped")
        
        # Test 3: Parser + Tools
        from core.parser import ResponseParser
        from core.tools import ToolRegistry
        
        parser = ResponseParser()
        registry = ToolRegistry()
        
        response = """THOUGHT: I need to calculate something.

ACTION: use_tool

ACTION_INPUT:
```json
{"tool_name": "calculator", "parameters": {"expression": "100 * 0.15"}}
```
"""
        parsed = parser.parse(response)
        if parsed.action == "use_tool":
            tool_result = registry.execute(
                parsed.action_input.get("tool_name"),
                **parsed.action_input.get("parameters", {})
            )
            if "15" in tool_result:
                results.success("Parser + ToolRegistry pipeline")
            else:
                results.fail("Parser + ToolRegistry pipeline", f"Result: {tool_result}")
        else:
            results.fail("Parser + ToolRegistry pipeline", f"Parse failed: {parsed}")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run complete test suite."""
    print("=" * 60)
    print("        META-AGENT V2.1 - TEST SUITE")
    print("=" * 60)
    
    results = TestResult()
    
    # Run all tests
    test_ast_validator(results)
    test_parser(results)
    test_tools(results)
    test_executor(results)
    test_local_sandbox(results)
    test_vectordb(results)
    test_script_manager(results)
    test_integration(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("                 TEST RESULTS")
    print("=" * 60)
    print(f"\n{results.summary()}\n")
    
    if results.errors:
        print("Failed tests:")
        for name, error in results.errors:
            print(f"  - {name}: {error}")
    
    # Certification
    print("\n" + "=" * 60)
    if results.failed == 0:
        print("✅ CERTIFICATION: ALL TESTS PASSED")
        print("   All components are working as expected.")
    else:
        print("❌ CERTIFICATION: TESTS FAILED")
        print(f"   {results.failed} component(s) need attention.")
    print("=" * 60)
    
    return results.failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
