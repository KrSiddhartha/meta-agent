"""
Meta-Agent v2 Examples

Practical examples demonstrating agent capabilities:
1. Basic tool usage
2. Tool creation and persistence
3. Code execution
4. Tool search and reuse
5. Complex multi-step tasks
6. vLLM integration
"""

import os
from agent import create_agent, MetaAgent


def example_basic_usage():
    """Example 1: Basic agent usage with existing tools."""
    print("\n" + "="*60)
    print("Example 1: Basic Tool Usage")
    print("="*60)
    
    agent = create_agent(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        provider="huggingface",
        storage_path="./example_data",
        use_docker=False,  # Use local sandbox for quick demo
    )
    
    # Simple calculation
    response = agent.run("Calculate the compound interest on 50000 at 8% for 5 years")
    print(f"Response: {response}")


def example_tool_creation():
    """Example 2: Create and persist a custom tool."""
    print("\n" + "="*60)
    print("Example 2: Tool Creation & Persistence")
    print("="*60)
    
    agent = create_agent(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        provider="huggingface",
        storage_path="./example_data",
        use_docker=False,
    )
    
    # Ask agent to create a tool
    response = agent.run("""
        Create a tool called 'roi_calculator' that calculates Return on Investment.
        Formula: ROI = ((Final Value - Initial Investment) / Initial Investment) * 100
        It should take initial_investment and final_value as parameters.
    """)
    print(f"Response: {response}")
    
    # Now use the created tool
    response2 = agent.run("Use the roi_calculator tool to calculate ROI if I invested 100000 and now have 150000")
    print(f"Using created tool: {response2}")


def example_tool_reuse():
    """Example 3: Tool persistence and reuse across sessions."""
    print("\n" + "="*60)
    print("Example 3: Tool Reuse Across Sessions")
    print("="*60)
    
    # First session - create tool
    agent1 = create_agent(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        provider="huggingface",
        storage_path="./example_data",
        use_docker=False,
    )
    
    agent1.run("""
        Create a tool called 'break_even_calculator' that calculates 
        break-even point. Formula: fixed_costs / (price_per_unit - variable_cost_per_unit)
    """)
    
    print("Tool created in session 1. Starting new session...")
    
    # Second session - tool should be available
    agent2 = create_agent(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        provider="huggingface",
        storage_path="./example_data",  # Same storage path
        use_docker=False,
    )
    
    # Search for the tool
    response = agent2.run("Search for tools related to break-even calculation")
    print(f"Tool search result: {response}")
    
    # Use the tool
    response2 = agent2.run("""
        Use break_even_calculator with fixed_costs=50000, 
        price_per_unit=100, variable_cost_per_unit=60
    """)
    print(f"Using persisted tool: {response2}")


def example_code_execution():
    """Example 4: Direct code execution for complex calculations."""
    print("\n" + "="*60)
    print("Example 4: Code Execution")
    print("="*60)
    
    agent = create_agent(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        provider="huggingface",
        storage_path="./example_data",
        use_docker=False,
    )
    
    response = agent.run("""
        Execute Python code to analyze this data and find the average, 
        max, and standard deviation:
        Sales data: [12500, 15000, 13200, 18000, 14500, 16800, 19200, 15500]
    """)
    print(f"Response: {response}")


def example_data_analysis_tools():
    """Example 5: Create data analysis tools."""
    print("\n" + "="*60)
    print("Example 5: Data Analysis Tools")
    print("="*60)
    
    agent = create_agent(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        provider="huggingface",
        storage_path="./example_data",
        use_docker=False,
    )
    
    # Create a moving average calculator
    response = agent.run("""
        Create a tool called 'moving_average' that calculates the 
        simple moving average of a list of numbers for a given window size.
        Parameters: data (list of numbers), window (integer)
    """)
    print(f"Tool creation: {response}")
    
    # Use it
    response2 = agent.run("""
        Calculate the 3-period moving average for: [10, 20, 30, 40, 50, 60, 70]
    """)
    print(f"Result: {response2}")


def example_vllm_usage():
    """Example 6: Using vLLM for faster local inference."""
    print("\n" + "="*60)
    print("Example 6: vLLM Integration")
    print("="*60)
    
    print("""
    To use vLLM:
    
    1. Start vLLM server:
       python -m vllm.entrypoints.openai.api_server \\
           --model mistralai/Mistral-7B-Instruct-v0.2 \\
           --dtype half \\
           --max-model-len 4096
    
    2. Create agent with vLLM provider:
    """)
    
    # Uncomment to use when vLLM is running:
    """
    agent = create_agent(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        provider="vllm",
        base_url="http://localhost:8000/v1",
        storage_path="./example_data",
        use_docker=True,
        docker_python_version="3.12",
    )
    
    response = agent.run("Calculate factorial of 10 using a custom tool")
    print(f"Response: {response}")
    """
    
    print("vLLM example code shown above (requires running vLLM server)")


def example_docker_sandbox():
    """Example 7: Docker-isolated execution."""
    print("\n" + "="*60)
    print("Example 7: Docker Sandbox")
    print("="*60)
    
    print("""
    Docker sandbox provides:
    - Isolated Python environment
    - Configurable Python version (3.9 - 3.12+)
    - Memory and CPU limits
    - Network isolation (optional)
    - Automatic cleanup
    
    Usage:
    """)
    
    # Uncomment when Docker is available:
    """
    agent = create_agent(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        provider="huggingface",
        storage_path="./example_data",
        use_docker=True,
        docker_python_version="3.12",
    )
    
    response = agent.run('''
        Execute code that:
        1. Imports numpy and pandas
        2. Creates a sample dataframe
        3. Calculates basic statistics
    ''')
    print(f"Response: {response}")
    """
    
    print("Docker example code shown above (requires Docker)")


def example_logistics_analysis():
    """Example 8: Logistics cost analysis."""
    print("\n" + "="*60)
    print("Example 8: Logistics Cost Analysis")
    print("="*60)
    
    agent = create_agent(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        provider="huggingface",
        storage_path="./example_data",
        use_docker=False,
    )
    
    # Create logistics tools
    response = agent.run("""
        Create a tool called 'shipping_cost_calculator' that calculates 
        total shipping cost. Formula: 
        base_rate + (weight_kg * rate_per_kg) + (distance_km * rate_per_km)
        
        Parameters: base_rate, weight_kg, rate_per_kg, distance_km, rate_per_km
    """)
    print(f"Tool creation: {response}")
    
    # Use the tool
    response2 = agent.run("""
        Calculate shipping cost for:
        - Base rate: 500
        - Weight: 25 kg at 10 per kg
        - Distance: 800 km at 0.5 per km
    """)
    print(f"Result: {response2}")


def example_financial_modeling():
    """Example 9: Financial modeling tools."""
    print("\n" + "="*60)
    print("Example 9: Financial Modeling")
    print("="*60)
    
    agent = create_agent(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        provider="huggingface",
        storage_path="./example_data",
        use_docker=False,
    )
    
    # Create NPV calculator
    response = agent.run("""
        Create a tool called 'npv_calculator' that calculates Net Present Value.
        It should take:
        - initial_investment: the upfront cost (negative)
        - cash_flows: list of future cash flows
        - discount_rate: annual discount rate (e.g., 0.10 for 10%)
        
        Formula: NPV = sum(CF_t / (1 + r)^t) for each period t
    """)
    print(f"Tool creation: {response}")
    
    # Use it
    response2 = agent.run("""
        Calculate NPV for a project with:
        - Initial investment: 100000
        - Cash flows for next 5 years: [30000, 35000, 40000, 45000, 50000]
        - Discount rate: 12%
    """)
    print(f"Result: {response2}")


def run_all_examples():
    """Run all examples."""
    print("\n" + "="*60)
    print("     META-AGENT V2 EXAMPLES")
    print("="*60)
    
    # Note: These require HF_TOKEN to be set
    if not os.environ.get("HF_TOKEN"):
        print("\n⚠️  Warning: HF_TOKEN not set. Set it with:")
        print("   export HF_TOKEN='your-huggingface-token'")
        print("\nShowing example code only...\n")
        return
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Tool Creation", example_tool_creation),
        ("Tool Reuse", example_tool_reuse),
        ("Code Execution", example_code_execution),
        ("Data Analysis", example_data_analysis_tools),
        ("vLLM Integration", example_vllm_usage),
        ("Docker Sandbox", example_docker_sandbox),
        ("Logistics Analysis", example_logistics_analysis),
        ("Financial Modeling", example_financial_modeling),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRun with: python examples.py <number>")
    print("Or run specific example: python -c \"from examples import example_basic_usage; example_basic_usage()\"")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        examples = [
            example_basic_usage,
            example_tool_creation,
            example_tool_reuse,
            example_code_execution,
            example_data_analysis_tools,
            example_vllm_usage,
            example_docker_sandbox,
            example_logistics_analysis,
            example_financial_modeling,
        ]
        
        try:
            idx = int(sys.argv[1]) - 1
            if 0 <= idx < len(examples):
                examples[idx]()
            else:
                print(f"Invalid example number. Choose 1-{len(examples)}")
        except ValueError:
            print("Usage: python examples.py <example_number>")
    else:
        run_all_examples()
