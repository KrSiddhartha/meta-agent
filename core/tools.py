"""
Tool System: Registry, Management, and Built-in Tools

Features:
- Tool registration and lookup
- Parameter validation
- Built-in utility tools
- Source code tracking for persistence
"""

import json
import math
import random
from typing import Any, Callable, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Tool:
    """Represents a callable tool."""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    source_code: Optional[str] = None  # Explicit source for persistence
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0
    
    def __call__(self, **kwargs) -> Any:
        """Execute the tool."""
        self.usage_count += 1
        return self.function(**kwargs)
    
    def to_dict(self) -> Dict:
        """Serialize tool metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "created_at": self.created_at,
            "usage_count": self.usage_count,
            "has_source": self.source_code is not None,
        }


class ToolRegistry:
    """
    Central registry for all available tools.
    
    Features:
    - Register/unregister tools
    - Lookup by name
    - List with descriptions
    - Parameter validation
    """
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_builtins()
    
    def _register_builtins(self):
        """Register built-in utility tools."""
        # Calculator
        self.register(Tool(
            name="calculator",
            description="Perform mathematical calculations. Supports basic operations and math functions.",
            function=self._builtin_calculator,
            parameters={
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2 * 3')",
                    "required": True
                }
            },
            source_code="# Built-in calculator tool"
        ))
        
        # Random number generator
        self.register(Tool(
            name="random_number",
            description="Generate a random number within a range.",
            function=self._builtin_random,
            parameters={
                "min": {"type": "number", "description": "Minimum value", "default": 0},
                "max": {"type": "number", "description": "Maximum value", "default": 100}
            },
            source_code="# Built-in random tool"
        ))
        
        # JSON formatter
        self.register(Tool(
            name="format_json",
            description="Format and validate JSON data.",
            function=self._builtin_format_json,
            parameters={
                "data": {"type": "any", "description": "Data to format as JSON", "required": True}
            },
            source_code="# Built-in JSON formatter"
        ))
        
        # List tools
        self.register(Tool(
            name="list_tools",
            description="List all available tools with their descriptions.",
            function=lambda: self.get_tools_description(),
            parameters={},
            source_code="# Built-in list tools"
        ))
    
    def _builtin_calculator(self, expression: str) -> str:
        """Safe calculator using limited eval."""
        allowed_names = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow,
            'sqrt': math.sqrt, 'log': math.log, 'log10': math.log10,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'pi': math.pi, 'e': math.e,
        }
        
        try:
            # Sanitize expression
            for char in expression:
                if char not in '0123456789+-*/.() ,':
                    if not char.isalpha():
                        raise ValueError(f"Invalid character: {char}")
            
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {e}"
    
    def _builtin_random(self, min: float = 0, max: float = 100) -> str:
        """Generate random number."""
        if isinstance(min, int) and isinstance(max, int):
            result = random.randint(int(min), int(max))
        else:
            result = random.uniform(float(min), float(max))
        return f"Random number: {result}"
    
    def _builtin_format_json(self, data: Any) -> str:
        """Format data as JSON."""
        try:
            if isinstance(data, str):
                data = json.loads(data)
            return json.dumps(data, indent=2)
        except Exception as e:
            return f"JSON error: {e}"
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
        print(f"[Tools] Registered: {tool.name}")
    
    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self.tools:
            del self.tools[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self.tools.get(name)
    
    def exists(self, name: str) -> bool:
        """Check if tool exists."""
        return name in self.tools
    
    def list_names(self) -> List[str]:
        """List all tool names."""
        return list(self.tools.keys())
    
    def list_tools(self) -> List[Tool]:
        """List all tools."""
        return list(self.tools.values())
    
    def get_tools_description(self) -> str:
        """Get formatted description of all tools."""
        if not self.tools:
            return "No tools available."
        
        descriptions = []
        for name, tool in sorted(self.tools.items()):
            param_str = ""
            if tool.parameters:
                params = []
                for pname, pinfo in tool.parameters.items():
                    ptype = pinfo.get("type", "any")
                    required = pinfo.get("required", False)
                    req_str = " (required)" if required else ""
                    params.append(f"  - {pname}: {ptype}{req_str}")
                param_str = "\n" + "\n".join(params)
            
            descriptions.append(f"• {name}: {tool.description}{param_str}")
        
        return "\n\n".join(descriptions)
    
    def execute(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name."""
        tool = self.get(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found. Available: {self.list_names()}"
        
        try:
            result = tool(**kwargs)
            return str(result) if result is not None else "Tool executed successfully."
        except Exception as e:
            return f"Tool execution error: {type(e).__name__}: {e}"
    
    def create_tool_from_code(
        self,
        name: str,
        description: str,
        code: str,
        parameters: Dict = None
    ) -> Optional[Tool]:
        """
        Create a new tool from code string.
        
        Args:
            name: Tool name
            description: Tool description
            code: Python code defining the tool function
            parameters: Parameter schema
        
        Returns:
            Created Tool or None if failed
        """
        # The code should define a function with the tool name
        namespace = {}
        
        try:
            exec(code, namespace)
            
            # Find the function
            func = namespace.get(name)
            if not callable(func):
                # Try to find any callable
                for key, val in namespace.items():
                    if callable(val) and not key.startswith('_'):
                        func = val
                        break
            
            if not callable(func):
                print(f"[Tools] No callable function found in code for '{name}'")
                return None
            
            tool = Tool(
                name=name,
                description=description,
                function=func,
                parameters=parameters or {},
                source_code=code  # Store original code
            )
            
            self.register(tool)
            return tool
            
        except Exception as e:
            print(f"[Tools] Failed to create tool '{name}': {e}")
            return None


class ToolResult:
    """Wrapper for tool execution results."""
    
    def __init__(self, success: bool, output: str, tool_name: str, execution_time_ms: float = 0):
        self.success = success
        self.output = output
        self.tool_name = tool_name
        self.execution_time_ms = execution_time_ms
    
    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"[{status}] {self.tool_name}: {self.output}"
