"""
Unified Code Executor: Docker-first with Local fallback.

Key insight: AST validation is only needed when Docker is unavailable.
Docker provides real isolation - code inside can do anything safely.

Permission Levels:
- RESTRICTED: Python-only, limited packages (default)
- ELEVATED: Can use subprocess, system calls, additional languages
- SYSTEM: Full host access (dangerous, requires explicit approval)
"""

import os
import uuid
import json
import time
import tempfile
import shutil
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


class PermissionLevel(Enum):
    """Execution permission levels."""
    RESTRICTED = "restricted"  # Safe, limited packages
    ELEVATED = "elevated"      # subprocess, docker-in-docker, multi-language
    SYSTEM = "system"          # Full host access (use with caution)


@dataclass
class ExecutionResult:
    """Result from code execution."""
    success: bool
    output: str
    error: Optional[str]
    return_value: Any
    execution_time_ms: float
    executor: str  # "docker" or "local"
    language: str  # "python", "rust", "bash"
    exit_code: int = 0


@dataclass 
class ExecutorConfig:
    """Configuration for code executor."""
    # Docker settings
    python_image: str = "python:3.12-slim"
    rust_image: str = "rust:1.75-slim"
    node_image: str = "node:20-slim"
    
    # Resource limits
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    timeout_seconds: int = 60
    
    # Network
    network_enabled: bool = False
    
    # Pre-installed packages per permission level
    restricted_packages: List[str] = field(default_factory=lambda: [
        "numpy", "pandas", "pyyaml", "requests"
    ])
    elevated_packages: List[str] = field(default_factory=lambda: [
        "numpy", "pandas", "pyyaml", "requests", 
        "scipy", "scikit-learn", "matplotlib"
    ])


class DockerExecutor:
    """
    Docker-based code execution.
    
    No AST validation - Docker IS the security boundary.
    Supports multiple languages via different images.
    """
    
    def __init__(self, config: ExecutorConfig = None):
        self.config = config or ExecutorConfig()
        self._client = None
        self._available = None
        
        self.work_dir = Path(tempfile.gettempdir()) / "meta_agent_executor"
        self.work_dir.mkdir(exist_ok=True)
    
    @property
    def available(self) -> bool:
        """Check if Docker is available."""
        if self._available is None:
            try:
                import docker
                client = docker.from_env()
                client.ping()
                self._client = client
                self._available = True
            except Exception:
                self._available = False
        return self._available
    
    @property
    def client(self):
        if not self._client:
            import docker
            self._client = docker.from_env()
        return self._client
    
    def execute_python(
        self,
        code: str,
        permission: PermissionLevel = PermissionLevel.RESTRICTED,
        inputs: Dict[str, Any] = None,
        packages: List[str] = None,
        timeout: int = None
    ) -> ExecutionResult:
        """Execute Python code in Docker."""
        
        exec_id = str(uuid.uuid4())[:8]
        exec_dir = self.work_dir / exec_id
        exec_dir.mkdir(exist_ok=True)
        
        try:
            # Write code
            code_file = exec_dir / "main.py"
            wrapper = self._python_wrapper(code, inputs)
            code_file.write_text(wrapper)
            
            # Write inputs
            if inputs:
                (exec_dir / "inputs.json").write_text(json.dumps(inputs))
            
            # Determine packages
            if packages is None:
                if permission == PermissionLevel.RESTRICTED:
                    packages = self.config.restricted_packages
                else:
                    packages = self.config.elevated_packages
            
            # Build command
            install_cmd = f"pip install -q {' '.join(packages)} && " if packages else ""
            command = f"/bin/sh -c '{install_cmd}python /workspace/main.py'"
            
            # Run
            return self._run_container(
                image=self.config.python_image,
                command=command,
                exec_dir=exec_dir,
                timeout=timeout or self.config.timeout_seconds,
                permission=permission,
                language="python"
            )
        finally:
            shutil.rmtree(exec_dir, ignore_errors=True)
    
    def execute_rust(
        self,
        code: str,
        permission: PermissionLevel = PermissionLevel.ELEVATED,
        cargo_deps: Dict[str, str] = None,
        timeout: int = None
    ) -> ExecutionResult:
        """Execute Rust code in Docker."""
        
        exec_id = str(uuid.uuid4())[:8]
        exec_dir = self.work_dir / exec_id
        exec_dir.mkdir(exist_ok=True)
        
        try:
            # Create Cargo project structure
            src_dir = exec_dir / "src"
            src_dir.mkdir()
            
            # Write main.rs
            (src_dir / "main.rs").write_text(code)
            
            # Write Cargo.toml
            cargo_deps = cargo_deps or {}
            deps_str = "\n".join(f'{k} = "{v}"' for k, v in cargo_deps.items())
            
            cargo_toml = f'''[package]
name = "agent_rust_exec"
version = "0.1.0"
edition = "2021"

[dependencies]
{deps_str}

[profile.release]
opt-level = 3
'''
            (exec_dir / "Cargo.toml").write_text(cargo_toml)
            
            # Build and run command
            command = "/bin/sh -c 'cd /workspace && cargo build --release 2>&1 && ./target/release/agent_rust_exec'"
            
            return self._run_container(
                image=self.config.rust_image,
                command=command,
                exec_dir=exec_dir,
                timeout=timeout or self.config.timeout_seconds * 2,  # Rust compilation takes longer
                permission=permission,
                language="rust"
            )
        finally:
            shutil.rmtree(exec_dir, ignore_errors=True)
    
    def execute_bash(
        self,
        script: str,
        permission: PermissionLevel = PermissionLevel.ELEVATED,
        timeout: int = None
    ) -> ExecutionResult:
        """Execute bash script in Docker."""
        
        exec_id = str(uuid.uuid4())[:8]
        exec_dir = self.work_dir / exec_id
        exec_dir.mkdir(exist_ok=True)
        
        try:
            # Write script
            script_file = exec_dir / "script.sh"
            script_file.write_text(script)
            
            command = "/bin/sh /workspace/script.sh"
            
            return self._run_container(
                image=self.config.python_image,  # Use Python image (has bash)
                command=command,
                exec_dir=exec_dir,
                timeout=timeout or self.config.timeout_seconds,
                permission=permission,
                language="bash"
            )
        finally:
            shutil.rmtree(exec_dir, ignore_errors=True)
    
    def _run_container(
        self,
        image: str,
        command: str,
        exec_dir: Path,
        timeout: int,
        permission: PermissionLevel,
        language: str
    ) -> ExecutionResult:
        """Run a Docker container."""
        
        start_time = time.time()
        
        try:
            # Network based on permission
            network_disabled = not self.config.network_enabled
            if permission == PermissionLevel.SYSTEM:
                network_disabled = False  # System level allows network
            
            container = self.client.containers.run(
                image=image,
                command=command,
                volumes={str(exec_dir): {"bind": "/workspace", "mode": "rw"}},
                mem_limit=self.config.memory_limit,
                cpu_period=100000,
                cpu_quota=int(self.config.cpu_limit * 100000),
                network_disabled=network_disabled,
                remove=False,
                detach=True,
            )
            
            # Wait with timeout
            try:
                result = container.wait(timeout=timeout)
                exit_code = result.get("StatusCode", -1)
            except Exception:
                container.kill()
                return ExecutionResult(
                    success=False,
                    output="",
                    error="Execution timed out",
                    return_value=None,
                    execution_time_ms=timeout * 1000,
                    executor="docker",
                    language=language,
                    exit_code=-1
                )
            
            # Get output
            stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
            stderr = container.logs(stdout=False, stderr=True).decode('utf-8')
            
            # Get return value if exists
            return_value = None
            result_file = exec_dir / "result.json"
            if result_file.exists():
                try:
                    return_value = json.loads(result_file.read_text())
                except:
                    pass
            
            container.remove(force=True)
            
            return ExecutionResult(
                success=exit_code == 0,
                output=stdout,
                error=stderr if stderr else None,
                return_value=return_value,
                execution_time_ms=(time.time() - start_time) * 1000,
                executor="docker",
                language=language,
                exit_code=exit_code
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Docker error: {type(e).__name__}: {e}",
                return_value=None,
                execution_time_ms=(time.time() - start_time) * 1000,
                executor="docker",
                language=language,
                exit_code=-1
            )
    
    def _python_wrapper(self, code: str, inputs: Dict = None) -> str:
        """Create Python wrapper that handles I/O."""
        return f'''
import json
import sys
from pathlib import Path

# Load inputs
inputs = {{}}
input_file = Path("/workspace/inputs.json")
if input_file.exists():
    inputs = json.loads(input_file.read_text())
    globals().update(inputs)

# Execute
__result__ = None
try:
    exec("""{code.replace('"""', chr(92) + '"""')}""")
    
    # Capture result
    for var in ['result', 'output', 'answer']:
        if var in dir():
            __result__ = eval(var)
            break
except Exception as e:
    print(f"Error: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    sys.exit(1)

# Save result
if __result__ is not None:
    try:
        Path("/workspace/result.json").write_text(json.dumps(__result__, default=str))
    except:
        pass
'''


class LocalExecutor:
    """
    Local execution fallback when Docker unavailable.
    
    THIS is where AST validation is required - no container isolation.
    """
    
    def __init__(self, timeout: int = 10):
        from .ast_validator import ASTSecurityValidator
        self.validator = ASTSecurityValidator()
        self.timeout = timeout
    
    def execute_python(
        self,
        code: str,
        permission: PermissionLevel = PermissionLevel.RESTRICTED,
        inputs: Dict[str, Any] = None,
        **kwargs
    ) -> ExecutionResult:
        """Execute Python locally with AST validation."""
        
        # ELEVATED/SYSTEM not allowed locally - too dangerous
        if permission != PermissionLevel.RESTRICTED:
            return ExecutionResult(
                success=False,
                output="",
                error="ELEVATED/SYSTEM permissions require Docker. Install Docker or use RESTRICTED.",
                return_value=None,
                execution_time_ms=0,
                executor="local",
                language="python"
            )
        
        # Validate code (required for local execution!)
        report = self.validator.validate(code)
        if not report.is_safe:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Security violation: {', '.join(report.violations)}",
                return_value=None,
                execution_time_ms=0,
                executor="local",
                language="python"
            )
        
        # Execute
        import sys
        from io import StringIO
        import signal
        
        start_time = time.time()
        
        namespace = {"__builtins__": __builtins__}
        if inputs:
            namespace["inputs"] = inputs  # Add inputs dict itself
            namespace.update(inputs)       # Also add individual values
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        # Timeout handler
        def timeout_handler(signum, frame):
            raise TimeoutError("Execution timed out")
        
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
        
        try:
            exec(code, namespace)
            output = sys.stdout.getvalue()
            
            return_value = None
            for var in ['result', 'output', 'answer']:
                if var in namespace:
                    return_value = namespace[var]
                    break
            
            return ExecutionResult(
                success=True,
                output=output,
                error=None,
                return_value=return_value,
                execution_time_ms=(time.time() - start_time) * 1000,
                executor="local",
                language="python"
            )
            
        except TimeoutError:
            return ExecutionResult(
                success=False,
                output=sys.stdout.getvalue(),
                error="Execution timed out",
                return_value=None,
                execution_time_ms=self.timeout * 1000,
                executor="local",
                language="python"
            )
        except Exception as e:
            import traceback
            return ExecutionResult(
                success=False,
                output=sys.stdout.getvalue(),
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                return_value=None,
                execution_time_ms=(time.time() - start_time) * 1000,
                executor="local",
                language="python"
            )
        finally:
            sys.stdout = old_stdout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
    
    def execute_rust(self, code: str, **kwargs) -> ExecutionResult:
        """Rust requires Docker."""
        return ExecutionResult(
            success=False,
            output="",
            error="Rust execution requires Docker. Please install Docker.",
            return_value=None,
            execution_time_ms=0,
            executor="local",
            language="rust"
        )
    
    def execute_bash(self, script: str, **kwargs) -> ExecutionResult:
        """Bash requires Docker for safety."""
        return ExecutionResult(
            success=False,
            output="",
            error="Bash execution requires Docker. Please install Docker.",
            return_value=None,
            execution_time_ms=0,
            executor="local",
            language="bash"
        )


class UnifiedExecutor:
    """
    Unified executor: Docker if available, Local fallback.
    
    Automatically chooses the right backend.
    """
    
    def __init__(self, config: ExecutorConfig = None, prefer_docker: bool = True):
        self.config = config or ExecutorConfig()
        self.prefer_docker = prefer_docker
        
        self._docker = DockerExecutor(self.config)
        self._local = LocalExecutor()
        
        if self._docker.available:
            print("[Executor] Docker available - full capabilities enabled")
        else:
            print("[Executor] Docker not available - using restricted local execution")
    
    @property
    def docker_available(self) -> bool:
        return self._docker.available
    
    def execute(
        self,
        code: str,
        language: str = "python",
        permission: PermissionLevel = PermissionLevel.RESTRICTED,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute code in the appropriate environment.
        
        Args:
            code: Source code
            language: "python", "rust", or "bash"
            permission: Permission level
            **kwargs: Language-specific options
        """
        
        # Route to appropriate executor
        if self.prefer_docker and self._docker.available:
            executor = self._docker
        else:
            executor = self._local
        
        # Route to language
        if language == "python":
            return executor.execute_python(code, permission=permission, **kwargs)
        elif language == "rust":
            return executor.execute_rust(code, permission=permission, **kwargs)
        elif language == "bash":
            return executor.execute_bash(code, permission=permission, **kwargs)
        else:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Unsupported language: {language}",
                return_value=None,
                execution_time_ms=0,
                executor="none",
                language=language
            )


# Factory function
def create_executor(config: ExecutorConfig = None) -> UnifiedExecutor:
    """Create unified executor."""
    return UnifiedExecutor(config)
