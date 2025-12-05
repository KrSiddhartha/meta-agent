"""
AST Security Validator: Pre-execution code analysis.

Uses BLOCKLIST approach (more practical than allowlist):
- Blocks dangerous modules: os, subprocess, sys, socket, etc.
- Blocks dangerous builtins: exec, eval, compile, open, etc.
- Allows useful modules: pandas, numpy, requests, json, etc.
"""

import ast
from typing import Set, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SecurityReport:
    """Report from security analysis."""
    is_safe: bool
    violations: List[str]
    warnings: List[str]
    blocked_imports: List[str]
    blocked_calls: List[str]
    
    def __str__(self) -> str:
        if self.is_safe:
            return "✓ Code passed security check"
        return f"✗ Security violations: {', '.join(self.violations)}"


class ASTSecurityValidator:
    """
    Analyze Python AST for security violations.
    
    Uses BLOCKLIST approach - blocks known dangerous modules/functions
    while allowing everything else (more practical for data science).
    """
    
    # Dangerous modules to block
    BLOCKED_MODULES: Set[str] = {
        # System access
        'os', 'sys', 'subprocess', 'shutil', 'pathlib',
        # Network (raw)
        'socket', 'socketserver', 'ssl',
        # Code execution
        'importlib', 'runpy', 'code', 'codeop',
        # Low-level
        'ctypes', 'cffi', 'mmap',
        # Serialization (can execute code)
        'pickle', 'shelve', 'marshal',
        # Multiprocessing (can spawn processes)
        'multiprocessing', 'concurrent',
        # Dangerous utilities
        'pty', 'tty', 'termios', 'fcntl',
        'resource', 'sysconfig', 'platform',
        # Web server
        'http.server', 'wsgiref', 'cgi',
    }
    
    # Dangerous built-in functions
    BLOCKED_BUILTINS: Set[str] = {
        'exec', 'eval', 'compile',
        'open', 'input',
        '__import__',
        'globals', 'locals', 'vars',
        'getattr', 'setattr', 'delattr',  # Can access private attributes
        'breakpoint',
    }
    
    # Dangerous attribute access
    BLOCKED_ATTRIBUTES: Set[str] = {
        '__class__', '__bases__', '__subclasses__',
        '__globals__', '__code__', '__builtins__',
        '__import__', '__loader__', '__spec__',
    }
    
    # Warnings (not blocked, but flagged)
    WARNING_MODULES: Set[str] = {
        'requests',  # Network calls
        'urllib',    # Network calls
        'asyncio',   # Can be complex
        'threading', # Concurrency
    }
    
    def __init__(
        self,
        extra_blocked_modules: Set[str] = None,
        extra_allowed_modules: Set[str] = None,
        strict_mode: bool = False
    ):
        """
        Initialize validator.
        
        Args:
            extra_blocked_modules: Additional modules to block
            extra_allowed_modules: Modules to explicitly allow (override blocklist)
            strict_mode: If True, also block warning modules
        """
        self.blocked_modules = self.BLOCKED_MODULES.copy()
        self.allowed_override: Set[str] = extra_allowed_modules or set()
        
        if extra_blocked_modules:
            self.blocked_modules.update(extra_blocked_modules)
        
        if strict_mode:
            self.blocked_modules.update(self.WARNING_MODULES)
    
    def validate(self, code: str) -> SecurityReport:
        """
        Validate code for security issues.
        
        Args:
            code: Python source code
            
        Returns:
            SecurityReport with analysis results
        """
        violations = []
        warnings = []
        blocked_imports = []
        blocked_calls = []
        
        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return SecurityReport(
                is_safe=False,
                violations=[f"Syntax error: {e}"],
                warnings=[],
                blocked_imports=[],
                blocked_calls=[]
            )
        
        # Walk the AST
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if self._is_blocked_module(module_name):
                        violations.append(f"Blocked import: {alias.name}")
                        blocked_imports.append(alias.name)
                    elif module_name in self.WARNING_MODULES:
                        warnings.append(f"Network/async module: {alias.name}")
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if self._is_blocked_module(module_name):
                        violations.append(f"Blocked import: from {node.module}")
                        blocked_imports.append(node.module)
                    elif module_name in self.WARNING_MODULES:
                        warnings.append(f"Network/async module: {node.module}")
            
            # Check function calls
            elif isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                if call_name:
                    if call_name in self.BLOCKED_BUILTINS:
                        violations.append(f"Blocked function: {call_name}()")
                        blocked_calls.append(call_name)
            
            # Check attribute access
            elif isinstance(node, ast.Attribute):
                if node.attr in self.BLOCKED_ATTRIBUTES:
                    violations.append(f"Blocked attribute: .{node.attr}")
            
            # Check string that looks like it might be trying to bypass
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(blocked in node.value for blocked in ['__import__', 'os.system', 'subprocess']):
                    warnings.append(f"Suspicious string constant detected")
        
        return SecurityReport(
            is_safe=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            blocked_imports=blocked_imports,
            blocked_calls=blocked_calls
        )
    
    def _is_blocked_module(self, module: str) -> bool:
        """Check if module is blocked."""
        if module in self.allowed_override:
            return False
        return module in self.blocked_modules
    
    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Extract function name from Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None
    
    def is_safe(self, code: str) -> Tuple[bool, str]:
        """
        Quick safety check.
        
        Returns:
            Tuple of (is_safe, message)
        """
        report = self.validate(code)
        if report.is_safe:
            return True, "Code is safe to execute"
        return False, f"Security violations: {', '.join(report.violations)}"


# ============================================================================
# Convenience Functions
# ============================================================================

_default_validator = None

def get_validator() -> ASTSecurityValidator:
    """Get default validator instance."""
    global _default_validator
    if _default_validator is None:
        _default_validator = ASTSecurityValidator()
    return _default_validator


def validate_code(code: str) -> SecurityReport:
    """Validate code using default validator."""
    return get_validator().validate(code)


def is_code_safe(code: str) -> bool:
    """Quick check if code is safe."""
    return get_validator().validate(code).is_safe
