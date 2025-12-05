"""
Grammar-Constrained Parser: Structured output parsing with outlines.

Features:
- Pydantic schema validation
- Outlines grammar-constrained generation
- Robust regex fallback
- JSON repair for malformed outputs
"""

import re
import json
from typing import Optional, Any, Dict
from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    """Valid agent actions."""
    USE_TOOL = "use_tool"
    CREATE_TOOL = "create_tool"
    EXECUTE_CODE = "execute_code"
    SEARCH_TOOLS = "search_tools"
    FINAL_ANSWER = "final_answer"


@dataclass
class ParsedResponse:
    """Structured response from LLM."""
    thought: str
    action: str
    action_input: Dict[str, Any]
    raw_response: str
    parse_method: str  # "outlines", "regex", "json_repair"
    
    @property
    def is_valid(self) -> bool:
        return bool(self.action)


class JSONRepairer:
    """Repair common JSON formatting issues from LLMs."""
    
    @staticmethod
    def repair(text: str) -> str:
        """Attempt to fix common JSON issues."""
        # Remove markdown code blocks
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        
        # Fix trailing commas
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        # Fix single quotes to double quotes (careful with apostrophes)
        # Only replace single quotes that look like JSON string delimiters
        text = re.sub(r"(?<![a-zA-Z])'([^']*)'(?![a-zA-Z])", r'"\1"', text)
        
        # Fix unquoted keys
        text = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
        
        # Fix Python None/True/False to JSON null/true/false
        text = re.sub(r'\bNone\b', 'null', text)
        text = re.sub(r'\bTrue\b', 'true', text)
        text = re.sub(r'\bFalse\b', 'false', text)
        
        return text.strip()
    
    @staticmethod
    def extract_json(text: str) -> Optional[Dict]:
        """Extract and parse JSON from text."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object
        patterns = [
            r'\{[^{}]*\}',  # Simple object
            r'\{(?:[^{}]|\{[^{}]*\})*\}',  # Nested one level
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    repaired = JSONRepairer.repair(match)
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    continue
        
        return None


class ResponseParser:
    """
    Parse LLM responses with multiple strategies.
    
    Priority:
    1. Outlines structured generation (if available)
    2. Regex extraction
    3. JSON repair fallback
    """
    
    # Regex patterns for ReAct format
    THOUGHT_PATTERN = re.compile(r"THOUGHT:\s*(.+?)(?=ACTION:|$)", re.DOTALL | re.IGNORECASE)
    ACTION_PATTERN = re.compile(r"ACTION:\s*(\w+)", re.IGNORECASE)
    ACTION_INPUT_PATTERN = re.compile(
        r"ACTION_INPUT:\s*```(?:json)?\s*(.+?)```",
        re.DOTALL | re.IGNORECASE
    )
    # Alternative without code blocks
    ACTION_INPUT_ALT_PATTERN = re.compile(
        r"ACTION_INPUT:\s*(\{.+?\})",
        re.DOTALL | re.IGNORECASE
    )
    
    def __init__(self, use_outlines: bool = True):
        """
        Initialize parser.
        
        Args:
            use_outlines: Whether to try outlines for structured generation
        """
        self.use_outlines = use_outlines
        self.outlines_available = self._check_outlines()
        
        if use_outlines and not self.outlines_available:
            print("[Parser] Outlines not available, using regex fallback")
    
    def _check_outlines(self) -> bool:
        """Check if outlines library is available."""
        try:
            import outlines
            return True
        except ImportError:
            return False
    
    def parse(self, response: str) -> ParsedResponse:
        """
        Parse LLM response into structured format.
        
        Args:
            response: Raw LLM output
            
        Returns:
            ParsedResponse with extracted components
        """
        # Try regex parsing first (most common case)
        result = self._parse_regex(response)
        if result.is_valid:
            return result
        
        # Try JSON repair
        result = self._parse_json_repair(response)
        if result.is_valid:
            return result
        
        # Return best effort
        return ParsedResponse(
            thought=response[:500],
            action="",
            action_input={},
            raw_response=response,
            parse_method="failed"
        )
    
    def _parse_regex(self, response: str) -> ParsedResponse:
        """Parse using regex patterns."""
        thought = ""
        action = ""
        action_input = {}
        
        # Extract THOUGHT
        thought_match = self.THOUGHT_PATTERN.search(response)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # Extract ACTION
        action_match = self.ACTION_PATTERN.search(response)
        if action_match:
            action = action_match.group(1).strip().lower()
        
        # Extract ACTION_INPUT
        input_match = self.ACTION_INPUT_PATTERN.search(response)
        if not input_match:
            input_match = self.ACTION_INPUT_ALT_PATTERN.search(response)
        
        if input_match:
            json_str = input_match.group(1).strip()
            try:
                action_input = json.loads(json_str)
            except json.JSONDecodeError:
                # Try repair
                repaired = JSONRepairer.repair(json_str)
                try:
                    action_input = json.loads(repaired)
                except json.JSONDecodeError:
                    action_input = {"raw": json_str}
        
        return ParsedResponse(
            thought=thought,
            action=action,
            action_input=action_input,
            raw_response=response,
            parse_method="regex"
        )
    
    def _parse_json_repair(self, response: str) -> ParsedResponse:
        """Try to extract any JSON from response."""
        extracted = JSONRepairer.extract_json(response)
        
        if extracted:
            return ParsedResponse(
                thought=extracted.get("thought", extracted.get("reasoning", "")),
                action=extracted.get("action", ""),
                action_input=extracted.get("action_input", extracted.get("input", {})),
                raw_response=response,
                parse_method="json_repair"
            )
        
        return ParsedResponse(
            thought="",
            action="",
            action_input={},
            raw_response=response,
            parse_method="json_repair_failed"
        )


class OutlinesGenerator:
    """
    Grammar-constrained generation using outlines library.
    
    This enforces output structure at generation time rather than parsing.
    """
    
    # JSON schema for agent response
    RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "Reasoning about the task"
            },
            "action": {
                "type": "string",
                "enum": ["use_tool", "create_tool", "execute_code", "search_tools", "final_answer"]
            },
            "action_input": {
                "type": "object",
                "description": "Parameters for the action"
            }
        },
        "required": ["thought", "action", "action_input"]
    }
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize outlines generator.
        
        Args:
            model_name: HuggingFace model to use
        """
        self.model_name = model_name
        self._model = None
        self._generator = None
    
    def _load_model(self):
        """Lazy load the outlines model."""
        if self._model is None:
            import outlines
            self._model = outlines.models.transformers(self.model_name)
            self._generator = outlines.generate.json(
                self._model,
                self.RESPONSE_SCHEMA
            )
    
    def generate(self, prompt: str) -> ParsedResponse:
        """
        Generate structured response using grammar constraints.
        
        Args:
            prompt: Input prompt
            
        Returns:
            ParsedResponse with guaranteed structure
        """
        self._load_model()
        
        try:
            result = self._generator(prompt)
            
            return ParsedResponse(
                thought=result.get("thought", ""),
                action=result.get("action", ""),
                action_input=result.get("action_input", {}),
                raw_response=json.dumps(result),
                parse_method="outlines"
            )
        except Exception as e:
            print(f"[Outlines] Generation failed: {e}")
            return ParsedResponse(
                thought="",
                action="",
                action_input={},
                raw_response=str(e),
                parse_method="outlines_failed"
            )


# ============================================================================
# Convenience Functions
# ============================================================================

def parse_response(response: str) -> ParsedResponse:
    """Quick parse function."""
    parser = ResponseParser(use_outlines=False)
    return parser.parse(response)


def create_parser(use_outlines: bool = False) -> ResponseParser:
    """Create parser instance."""
    return ResponseParser(use_outlines=use_outlines)
