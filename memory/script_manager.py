"""
Script Manager: Persist and load tools as Python scripts.

Features:
- Save tools with explicit source code (no inspect.getsource)
- Load tools from disk
- Index for quick lookup
- Capability extraction
"""

import os
import json
import importlib.util
from typing import Optional, Dict, List, Callable
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ToolMetadata:
    """Metadata for a persisted tool."""
    name: str
    description: str
    capabilities: List[str]
    parameters: Dict
    created_at: str
    file_path: str
    version: str = "1.0"


class ScriptManager:
    """
    Manages persistence of tools as Python scripts.
    
    Key fix: Uses explicit source code strings instead of inspect.getsource()
    which fails for dynamically created functions.
    """
    
    # Template for generating tool scripts
    TOOL_TEMPLATE = '''"""
Auto-generated Tool: {name}
Description: {description}
Capabilities: {capabilities}
Created: {created_at}
Version: {version}
"""

{source_code}

# Tool metadata for reconstruction
__tool_metadata__ = {{
    "name": "{name}",
    "description": "{description}",
    "capabilities": {capabilities},
    "parameters": {parameters},
    "version": "{version}",
    "created_at": "{created_at}"
}}

def get_tool():
    """Return the tool function."""
    return {function_name}

def get_metadata():
    """Return tool metadata."""
    return __tool_metadata__
'''
    
    def __init__(
        self,
        base_path: str = "./agent_scripts",
        vector_store = None
    ):
        """
        Initialize script manager.
        
        Args:
            base_path: Directory for storing scripts
            vector_store: Optional MilvusVectorStore for indexing
        """
        self.base_path = Path(base_path)
        self.tools_path = self.base_path / "tools"
        self.index_path = self.base_path / "index.json"
        
        # Create directories
        self.tools_path.mkdir(parents=True, exist_ok=True)
        
        # Vector store for semantic search
        self.vector_store = vector_store
        
        # Load index
        self.index = self._load_index()
        
        print(f"[ScriptManager] Initialized: {self.base_path}")
        print(f"[ScriptManager] Tools: {len(self.index.get('tools', {}))}")
    
    def _load_index(self) -> Dict:
        """Load the index file."""
        if self.index_path.exists():
            with open(self.index_path) as f:
                return json.load(f)
        return {"tools": {}, "agents": {}}
    
    def _save_index(self):
        """Save the index file."""
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2)
    
    def _extract_capabilities(self, description: str, code: str) -> List[str]:
        """Extract capabilities from description and code."""
        capabilities = []
        
        # Keywords to look for
        keyword_map = {
            "calculate": "mathematical_computation",
            "compute": "mathematical_computation",
            "math": "mathematical_computation",
            "analyze": "data_analysis",
            "parse": "data_processing",
            "transform": "data_transformation",
            "convert": "data_conversion",
            "format": "formatting",
            "validate": "validation",
            "search": "search",
            "filter": "filtering",
            "sort": "sorting",
            "aggregate": "aggregation",
            "api": "api_integration",
            "http": "http_requests",
            "file": "file_operations",
            "json": "json_processing",
            "csv": "csv_processing",
            "date": "date_handling",
            "time": "time_handling",
            "string": "string_manipulation",
            "text": "text_processing",
            "regex": "pattern_matching",
        }
        
        text = (description + " " + code).lower()
        
        for keyword, capability in keyword_map.items():
            if keyword in text and capability not in capabilities:
                capabilities.append(capability)
        
        return capabilities if capabilities else ["general"]
    
    def _extract_function_name(self, code: str) -> str:
        """Extract the main function name from code."""
        import re
        
        # Look for def statements
        matches = re.findall(r'def\s+(\w+)\s*\(', code)
        
        if matches:
            # Prefer functions that don't start with underscore
            for name in matches:
                if not name.startswith('_'):
                    return name
            return matches[0]
        
        return "tool_function"
    
    def save_tool(
        self,
        name: str,
        description: str,
        source_code: str,
        parameters: Dict = None,
        version: str = "1.0"
    ) -> str:
        """
        Save a tool to disk.
        
        Args:
            name: Tool name
            description: Tool description
            source_code: The actual source code (not from inspect)
            parameters: Parameter schema
            version: Version string
        
        Returns:
            Path to saved script
        """
        # Clean source code
        source_code = source_code.strip()
        if source_code.startswith("```"):
            # Remove markdown code blocks
            lines = source_code.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            source_code = "\n".join(lines)
        
        # Extract capabilities
        capabilities = self._extract_capabilities(description, source_code)
        
        # Extract function name
        function_name = self._extract_function_name(source_code)
        
        # Generate script content
        created_at = datetime.now().isoformat()
        
        script_content = self.TOOL_TEMPLATE.format(
            name=name,
            description=description,
            capabilities=capabilities,
            parameters=json.dumps(parameters or {}),
            source_code=source_code,
            function_name=function_name,
            version=version,
            created_at=created_at
        )
        
        # Save to file
        safe_name = name.replace(" ", "_").replace("-", "_").lower()
        file_path = self.tools_path / f"{safe_name}.py"
        
        with open(file_path, "w") as f:
            f.write(script_content)
        
        # Update index
        self.index["tools"][name] = {
            "name": name,
            "description": description,
            "capabilities": capabilities,
            "parameters": parameters or {},
            "file_path": str(file_path),
            "version": version,
            "created_at": created_at,
            "function_name": function_name
        }
        self._save_index()
        
        # Index in vector store if available
        if self.vector_store:
            from .vectordb import ToolRecord
            tool_record = ToolRecord(
                name=name,
                description=description,
                source_code=source_code,
                capabilities=capabilities,
                parameters=parameters or {},
            )
            self.vector_store.register_tool(tool_record)
        
        print(f"[ScriptManager] Saved tool: {name} -> {file_path}")
        return str(file_path)
    
    def load_tool(self, name: str) -> Optional[Callable]:
        """
        Load a tool from disk.
        
        Args:
            name: Tool name
        
        Returns:
            The tool function or None if not found
        """
        if name not in self.index.get("tools", {}):
            print(f"[ScriptManager] Tool not found: {name}")
            return None
        
        tool_info = self.index["tools"][name]
        file_path = tool_info["file_path"]
        
        if not os.path.exists(file_path):
            print(f"[ScriptManager] Tool file missing: {file_path}")
            return None
        
        try:
            # Load module dynamically
            spec = importlib.util.spec_from_file_location(name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the tool function
            if hasattr(module, 'get_tool'):
                return module.get_tool()
            
            # Fallback: try to get function by name
            function_name = tool_info.get("function_name", name)
            if hasattr(module, function_name):
                return getattr(module, function_name)
            
            print(f"[ScriptManager] Could not find function in {file_path}")
            return None
            
        except Exception as e:
            print(f"[ScriptManager] Error loading tool {name}: {e}")
            return None
    
    def get_tool_info(self, name: str) -> Optional[Dict]:
        """Get tool metadata without loading."""
        return self.index.get("tools", {}).get(name)
    
    def list_tools(self) -> List[str]:
        """List all saved tool names."""
        return list(self.index.get("tools", {}).keys())
    
    def search_tools(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search tools by capability or description.
        
        Uses vector store if available, otherwise keyword search.
        """
        if self.vector_store:
            return self.vector_store.search_tools(query, limit=limit)
        
        # Fallback: keyword search (check each word)
        results = []
        query_words = query.lower().split()
        
        for name, info in self.index.get("tools", {}).items():
            score = 0
            
            # Check name
            name_lower = name.lower()
            for word in query_words:
                if word in name_lower:
                    score += 0.5
            
            # Check description
            desc_lower = info.get("description", "").lower()
            for word in query_words:
                if word in desc_lower:
                    score += 0.3
            
            # Check capabilities
            caps_str = " ".join(info.get("capabilities", [])).lower()
            for word in query_words:
                if word in caps_str:
                    score += 0.2
            
            if score > 0:
                results.append({**info, "score": score})
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def delete_tool(self, name: str) -> bool:
        """Delete a tool."""
        if name not in self.index.get("tools", {}):
            return False
        
        tool_info = self.index["tools"][name]
        file_path = tool_info["file_path"]
        
        # Delete file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Remove from index
        del self.index["tools"][name]
        self._save_index()
        
        print(f"[ScriptManager] Deleted tool: {name}")
        return True
    
    def get_all_metadata(self) -> Dict[str, ToolMetadata]:
        """Get metadata for all tools."""
        result = {}
        for name, info in self.index.get("tools", {}).items():
            result[name] = ToolMetadata(
                name=info["name"],
                description=info["description"],
                capabilities=info["capabilities"],
                parameters=info["parameters"],
                created_at=info["created_at"],
                file_path=info["file_path"],
                version=info.get("version", "1.0")
            )
        return result
