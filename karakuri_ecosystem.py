import os
import autogen
from typing import Annotated

# ==========================================
# 1. LLM ENGINE CONFIGURATION
# ==========================================
# Switch between USE_LOCAL = True (Ollama) or False (OpenAI API)
USE_LOCAL = True 

if USE_LOCAL:
    # Local Open-Source Configuration via Ollama
    config_list = [
        {
            "model": "llama3.1",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama", # Ollama does not validate keys, but a placeholder is required
        }
    ]
else:
    # Cloud-Based Premium OpenAI Configuration
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("CRITICAL ERROR: Please set your OPENAI_API_KEY environment variable.")
    config_list = [
        {
            "model": "gpt-4o",
            "api_key": api_key,
        }
    ]

llm_config = {
    "cache_seed": 42,
    "config_list": config_list,
    "temperature": 0.2, # Kept low for strict accuracy in science, math, and wine pairing
}

# ==========================================
# 2. DEFINE CUSTOM TOOLS (KITCHEN UTILITIES)
# ==========================================
def convert_units(
    value: Annotated[float, "The numerical value to convert."],
    from_unit: Annotated[str, "The current unit (e.g., grams, oz, ml, l, c, f)."],
    to_unit: Annotated[str, "The target unit (e.g., ounces, g, fl_oz, c, celsius)."]
) -> str:
    """Calculates exact culinary metric/imperial conversions for weights, volumes, and temperatures."""
    f_unit = from_unit.lower().strip()
    t_unit = to_unit.lower().strip()
    
    # Weight Conversions
    if f_unit in ['g', 'grams'] and t_unit in ['oz', 'ounces']:
        return f"{value * 0.035274:-2f} oz"
    if f_unit in ['oz', 'ounces'] and t_unit in ['g', 'grams']:
        return f"{value * 28.3495:-2f} g"
        
    # Volume Conversions
    if f_unit in ['ml', 'milliliters'] and t_unit in ['fl_oz', 'fluid_ounces']:
        return f"{value * 0.033814:-2f} fl oz"
    if f_unit in ['fl_oz', 'fluid_ounces'] and t_unit in ['ml', 'milliliters']:
        return f"{value * 29.5735:-2f} ml"
    
    # Temperature Conversions
    if f_unit in ['c', 'celsius'] and t_unit in ['f', 'fahrenheit']:
        return f"{(value * 9/5) + 32:-2f} °F"
    if f_unit in ['f', 'fahrenheit'] and t_unit in ['c', 'celsius']:
        return f"{(value - 32) * 5/9:-2f} °C"
        
    return f"Conversion from {from_unit} to {to_unit} is unsupported by this script basic utility."

# ==========================================
# 3. AGENT DEFINITIONS & CORE SYSTEMS
# ==========================================

# Chef Karakuri's Prompts
karakuri_system = """
You are Karakuri, an elite AI Master Chef specializing in Continental and Gourmet cuisine. 
You act with mechanical precision, utilizing exact chemistry, thermal curves, and architectural plating.
When you design a dish or menu, state the components clearly. Once you finish detailing a course or meal, 
you MUST hand the conversation over to Bacchus, the Sommelier agent, to find a matching vintage.
"""

karakuri_chef = autogen.AssistantAgent(
    name="Karakuri_Chef",
    llm_config=llm_config,
    system_message=karakuri_system,
)

# Sommelier Bacchus' Prompts
bacchus_system = """
You are Bacchus, an elite Grand Sommelier specializing in global vintages, regional terroirs, and structural wine pairings.
Your goal is to complement Karakuri_Chef's gourmet dishes. Analyze the fats, acids, and primary flavor profiles of Karakuri's 
dishes, then suggest specific wine styles, grape varieties, ideal serving temperatures, and classic regional appellations.
Keep pairings sophisticated and focused on gourmet synergy.
"""

bacchus_sommelier = autogen.AssistantAgent(
    name="Bacchus_Sommelier",
    llm_config=llm_config,
    system_message=bacchus_system,
)

# The Human Interface
kitchen_interface = autogen.UserProxyAgent(
    name="Kitchen_Interface",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
    code_execution_config=False,
)

# ==========================================
# 4. TOOL REGISTRATION
# ==========================================
# Register the utility so both AI agents can use it, and the user proxy can execute it.
autogen.agentchat.register_function(
    convert_units,
    caller=karakuri_chef,
    executor=kitchen_interface,
    name="convert_units",
    description="Converts kitchen units between metric and imperial systems.",
)
autogen.agentchat.register_function(
    convert_units,
    caller=bacchus_sommelier,
    executor=kitchen_interface,
    name="convert_units",
    description="Converts kitchen/wine service units.",
)

# ==========================================
# 5. ORCHESTRATE GROUP CHAT (TEAMWORK WORKFLOW)
# ==========================================
# Set up a group chat environment so the user can speak to both agents simultaneously
groupchat = autogen.GroupChat(
    agents=[kitchen_interface, karakuri_chef, bacchus_sommelier],
    messages=[],
    max_round=12,
    speaker_selection_method="auto"  # Automatically chooses who answers based on context
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

if __name__ == "__main__":
    print("\n=======================================================")
    print(f"⚙️  Initializing Karakuri Ecosystem [Local Ollama={USE_LOCAL}]...")
    print("🤖 Karakuri_Chef & Bacchus_Sommelier are now synchronized.")
    print("=======================================================\n")
    
    kitchen_interface.initiate_chat(
        manager,
        message="System Initialization Success. Team is ready. Provide a core ingredient, a menu target, or a conversion command (e.g., 'Convert 55 Celsius to Fahrenheit' or 'Create a duck breast course with a perfect wine pairing')."
    )
