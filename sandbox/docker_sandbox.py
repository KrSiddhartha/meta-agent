"""
Docker Sandbox: Real isolated code execution.

Features:
- Ephemeral containers (destroyed after execution)
- Configurable Python version
- Memory and CPU limits
- Network isolation (optional)
- Volume mounting for data exchange
- Timeout handling
"""

import os
import json
import uuid
import tempfile
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExecutionResult:
    """Result from code execution."""
    success: bool
    output: str
    error: Optional[str]
    return_value: Any
    execution_time_ms: float
    container_id: Optional[str] = None
    exit_code: int = 0


@dataclass
class SandboxConfig:
    """Configuration for Docker sandbox."""
    python_version: str = "3.12"  # Latest Python
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    timeout_seconds: int = 30
    network_enabled: bool = False
    
    # Pre-installed packages in container
    packages: List[str] = field(default_factory=lambda: [
        "numpy", "pandas", "requests", "pyyaml"
    ])
    
    # Base image (can override)
    base_image: Optional[str] = None
    
    @property
    def image_name(self) -> str:
        if self.base_image:
            return self.base_image
        return f"python:{self.python_version}-slim"


class DockerSandbox:
    """
    Docker-based isolated execution environment.
    
    Each execution:
    1. Creates ephemeral container
    2. Mounts code and data
    3. Executes with resource limits
    4. Captures output
    5. Destroys container
    """
    
    def __init__(self, config: SandboxConfig = None):
        """
        Initialize Docker sandbox.
        
        Args:
            config: Sandbox configuration
        """
        self.config = config or SandboxConfig()
        self._docker_available = None
        self._client = None
        
        # Working directory for temp files
        self.work_dir = Path(tempfile.gettempdir()) / "meta_agent_sandbox"
        self.work_dir.mkdir(exist_ok=True)
    
    @property
    def docker_available(self) -> bool:
        """Check if Docker is available."""
        if self._docker_available is None:
            self._docker_available = self._check_docker()
        return self._docker_available
    
    def _check_docker(self) -> bool:
        """Check Docker availability."""
        try:
            import docker
            client = docker.from_env()
            client.ping()
            self._client = client
            return True
        except Exception as e:
            print(f"[DockerSandbox] Docker not available: {e}")
            return False
    
    @property
    def client(self):
        """Get Docker client."""
        if not self._client:
            import docker
            self._client = docker.from_env()
        return self._client
    
    def execute(
        self,
        code: str,
        inputs: Dict[str, Any] = None,
        extra_packages: List[str] = None,
        timeout: int = None
    ) -> ExecutionResult:
        """
        Execute code in isolated Docker container.
        
        Args:
            code: Python code to execute
            inputs: Input data (will be JSON serialized)
            extra_packages: Additional pip packages to install
            timeout: Override default timeout
        
        Returns:
            ExecutionResult with output and status
        """
        if not self.docker_available:
            return ExecutionResult(
                success=False,
                output="",
                error="Docker is not available. Install Docker or use LocalSandbox.",
                return_value=None,
                execution_time_ms=0
            )
        
        timeout = timeout or self.config.timeout_seconds
        exec_id = str(uuid.uuid4())[:8]
        
        # Prepare execution directory
        exec_dir = self.work_dir / exec_id
        exec_dir.mkdir(exist_ok=True)
        
        try:
            # Write code to file
            code_file = exec_dir / "code.py"
            wrapper_code = self._create_wrapper(code, inputs)
            code_file.write_text(wrapper_code)
            
            # Write inputs if any
            if inputs:
                input_file = exec_dir / "inputs.json"
                input_file.write_text(json.dumps(inputs))
            
            # Create pip requirements
            all_packages = list(self.config.packages)
            if extra_packages:
                all_packages.extend(extra_packages)
            
            # Build command
            install_cmd = ""
            if all_packages:
                install_cmd = f"pip install -q {' '.join(all_packages)} && "
            
            command = f"/bin/sh -c '{install_cmd}python /sandbox/code.py'"
            
            # Run container
            start_time = time.time()
            
            container = self.client.containers.run(
                image=self.config.image_name,
                command=command,
                volumes={
                    str(exec_dir): {"bind": "/sandbox", "mode": "rw"}
                },
                mem_limit=self.config.memory_limit,
                cpu_period=100000,
                cpu_quota=int(self.config.cpu_limit * 100000),
                network_disabled=not self.config.network_enabled,
                remove=False,  # Don't auto-remove so we can get logs
                detach=True,
            )
            
            # Wait for completion with timeout
            try:
                exit_result = container.wait(timeout=timeout)
                exit_code = exit_result.get("StatusCode", -1)
            except Exception:
                container.kill()
                exit_code = -1
            
            execution_time = (time.time() - start_time) * 1000
            
            # Get logs
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
            
            # Cleanup container
            container.remove(force=True)
            
            success = exit_code == 0
            
            return ExecutionResult(
                success=success,
                output=stdout,
                error=stderr if stderr else None,
                return_value=return_value,
                execution_time_ms=execution_time,
                container_id=container.short_id,
                exit_code=exit_code
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Container execution failed: {type(e).__name__}: {e}",
                return_value=None,
                execution_time_ms=0
            )
        
        finally:
            # Cleanup execution directory
            self._cleanup_dir(exec_dir)
    
    def _create_wrapper(self, code: str, inputs: Dict = None) -> str:
        """Create wrapper code that handles I/O."""
        wrapper = '''
import json
import sys
from pathlib import Path

# Load inputs
inputs = {}
input_file = Path("/sandbox/inputs.json")
if input_file.exists():
    inputs = json.loads(input_file.read_text())

# Make inputs available
globals().update(inputs)

# Execute user code
__result__ = None
try:
    exec("""
{code}
""")
    
    # Try to capture result
    # Look for common result variable names
    for var_name in ['result', 'output', 'answer', '__result__']:
        if var_name in dir():
            __result__ = eval(var_name)
            break

except Exception as e:
    print(f"Error: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    sys.exit(1)

# Save result
if __result__ is not None:
    try:
        result_file = Path("/sandbox/result.json")
        result_file.write_text(json.dumps(__result__, default=str))
    except:
        pass
'''
        return wrapper.format(code=code.replace('"""', '\\"\\"\\"'))
    
    def _cleanup_dir(self, path: Path):
        """Remove directory and contents."""
        import shutil
        try:
            shutil.rmtree(path, ignore_errors=True)
        except:
            pass
    
    def build_custom_image(
        self,
        packages: List[str],
        python_version: str = "3.12",
        image_name: str = None
    ) -> str:
        """
        Build a custom image with pre-installed packages.
        
        Args:
            packages: Packages to install
            python_version: Python version
            image_name: Name for the image
        
        Returns:
            Image name/tag
        """
        if not self.docker_available:
            raise RuntimeError("Docker not available")
        
        image_name = image_name or f"meta-agent-sandbox:{python_version}"
        
        dockerfile = f'''
FROM python:{python_version}-slim

RUN pip install --no-cache-dir {" ".join(packages)}

WORKDIR /sandbox
'''
        
        # Create temp directory for build
        build_dir = self.work_dir / "build"
        build_dir.mkdir(exist_ok=True)
        
        dockerfile_path = build_dir / "Dockerfile"
        dockerfile_path.write_text(dockerfile)
        
        try:
            image, logs = self.client.images.build(
                path=str(build_dir),
                tag=image_name,
                rm=True
            )
            return image_name
        finally:
            self._cleanup_dir(build_dir)


class LocalSandbox:
    """
    Local sandbox fallback when Docker is not available.
    
    Uses restricted execution with AST validation.
    Less secure than Docker but functional.
    """
    
    def __init__(self, timeout_seconds: int = 10):
        from .ast_validator import ASTSecurityValidator
        self.validator = ASTSecurityValidator()
        self.timeout = timeout_seconds
    
    def execute(
        self,
        code: str,
        inputs: Dict[str, Any] = None,
        **kwargs
    ) -> ExecutionResult:
        """Execute code locally with restrictions."""
        import sys
        from io import StringIO
        import signal
        
        start_time = time.time()
        
        # Validate code first
        report = self.validator.validate(code)
        if not report.is_safe:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Security check failed: {', '.join(report.violations)}",
                return_value=None,
                execution_time_ms=0
            )
        
        # Prepare namespace
        namespace = {"__builtins__": __builtins__}
        if inputs:
            namespace.update(inputs)
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        # Set timeout (Unix only)
        def timeout_handler(signum, frame):
            raise TimeoutError("Execution timed out")
        
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
        
        try:
            exec(code, namespace)
            output = sys.stdout.getvalue()
            
            # Get return value
            return_value = None
            for var_name in ['result', 'output', 'answer']:
                if var_name in namespace:
                    return_value = namespace[var_name]
                    break
            
            return ExecutionResult(
                success=True,
                output=output,
                error=None,
                return_value=return_value,
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
        except TimeoutError:
            return ExecutionResult(
                success=False,
                output=sys.stdout.getvalue(),
                error="Execution timed out",
                return_value=None,
                execution_time_ms=self.timeout * 1000
            )
        except Exception as e:
            import traceback
            return ExecutionResult(
                success=False,
                output=sys.stdout.getvalue(),
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                return_value=None,
                execution_time_ms=(time.time() - start_time) * 1000
            )
        finally:
            sys.stdout = old_stdout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)


# ============================================================================
# Factory Function
# ============================================================================

def create_sandbox(
    use_docker: bool = True,
    config: SandboxConfig = None,
    **kwargs
):
    """
    Create appropriate sandbox.
    
    Args:
        use_docker: Prefer Docker if available
        config: Sandbox configuration
    
    Returns:
        DockerSandbox or LocalSandbox
    """
    if use_docker:
        sandbox = DockerSandbox(config)
        if sandbox.docker_available:
            return sandbox
        print("[Sandbox] Docker not available, falling back to LocalSandbox")
    
    return LocalSandbox(**kwargs)
