import os
import autogen
from typing import Annotated
import gradio as gr

# ==========================================
# 1. LLM ENGINE & UTILITY TOOL CONFIGURATION
# ==========================================
USE_LOCAL = True 

if USE_LOCAL:
    config_list = [{
        "model": "llama3.1",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    }]
else:
    config_list = [{
        "model": "gpt-4o",
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
    }]

llm_config = {
    "cache_seed": None,  # Disabled for dynamic web outputs
    "config_list": config_list,
    "temperature": 0.2,
}

def convert_units(
    value: Annotated[float, "The numerical value to convert."],
    from_unit: Annotated[str, "The current unit (e.g., g, oz, ml, c, f)."],
    to_unit: Annotated[str, "The target unit (e.g., oz, g, fl_oz, f, c)."]
) -> str:
    """Calculates exact culinary metric/imperial conversions."""
    f_unit = from_unit.lower().strip()
    t_unit = to_unit.lower().strip()
    
    if f_unit in ['g', 'grams'] and t_unit in ['oz', 'ounces']:
        return f"{value * 0.035274:.2f} oz"
    if f_unit in ['oz', 'ounces'] and t_unit in ['g', 'grams']:
        return f"{value * 28.3495:.2f} g"
    if f_unit in ['ml', 'milliliters'] and t_unit in ['fl_oz', 'fluid_ounces']:
        return f"{value * 0.033814:.2f} fl oz"
    if f_unit in ['fl_oz', 'fluid_ounces'] and t_unit in ['ml', 'milliliters']:
        return f"{value * 29.5735:.2f} ml"
    if f_unit in ['c', 'celsius'] and t_unit in ['f', 'fahrenheit']:
        return f"{(value * 9/5) + 32:.2f} °F"
    if f_unit in ['f', 'fahrenheit'] and t_unit in ['c', 'celsius']:
        return f"{(value - 32) * 5/9:.2f} °C"
        
    return f"Conversion from {from_unit} to {to_unit} is unsupported."

# ==========================================
# 2. GRADIO HOOK FOR AUTOGEN AGENTS
# ==========================================
# We wrap the UI interaction into a function that captures individual agent utterances.
def run_culinary_session(user_message, history):
    chat_history = []
    
    # 1. Initialize Agents inside the session to prevent cross-talk
    karakuri_chef = autogen.AssistantAgent(
        name="🧑‍🍳 Karakuri_Chef",
        llm_config=llm_config,
        system_message="You are Karakuri, an elite AI Master Chef. Speak with absolute mechanical precision, structural chemistry, and exact temperatures. Once you finish detailing a course or solving a cooking technique problem, you MUST hand the conversation over to 🍷 Bacchus_Sommelier to discover a matching wine vintage.",
    )

    bacchus_sommelier = autogen.AssistantAgent(
        name="🍷 Bacchus_Sommelier",
        llm_config=llm_config,
        system_message="You are Bacchus, an elite Grand Sommelier specializing in global vintages and structural wine pairings. Analyze Karakuri's gourmet creations and suggest specific old-world or new-world wine styles, terroirs, and serving conditions.",
    )

    kitchen_interface = autogen.UserProxyAgent(
        name="Kitchen_Interface",
        human_input_mode="NEVER",  # Switched to NEVER so the UI orchestrates automatically
        max_consecutive_auto_reply=3,
        code_execution_config=False,
    )

    # Register tool execution capabilities
    autogen.agentchat.register_function(convert_units, caller=karakuri_chef, executor=kitchen_interface, name="convert_units", description="Converts weight, volume, or temperature units.")
    autogen.agentchat.register_function(convert_units, caller=bacchus_sommelier, executor=kitchen_interface, name="convert_units", description="Converts weight, volume, or temperature units.")

    groupchat = autogen.GroupChat(
        agents=[kitchen_interface, karakuri_chef, bacchus_sommelier],
        messages=[],
        max_round=5,
        speaker_selection_method="auto"
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # 2. Trigger the chat conversation
    kitchen_interface.initiate_chat(manager, message=user_message, clear_history=True)

    # 3. Process log entries to map them straight into HTML Chatbot components
    for msg in groupchat.messages:
        sender = msg['name']
        content = msg['content']
        
        # Format tool calls clean if they present inside the chat logs
        if not content and "tool_calls" in msg:
            content = f"⚙️ *System calling tool: convert_units() to compute metrics...*"
            
        if sender == "Kitchen_Interface":
            # Show what the user requested
            chat_history.append({"role": "user", "content": content})
        else:
            # Show which specific AI agent replied with custom avatars or prefixes
            prefix = "🧑‍🍳 **Karakuri (Chef):**\n\n" if "Chef" in sender else "🍷 **Bacchus (Sommelier):**\n\n"
            chat_history.append({"role": "assistant", "content": f"{prefix}{content}"})

    return "", chat_history

# ==========================================
# 3. HTML / CSS UI APP DESIGN
# ==========================================
custom_css = """
footer {visibility: hidden}
.gradio-container {background-color: #0f1115 !important; color: #e2e8f0;}
#title-header {text-align: center; margin-bottom: 20px;}
#title-header h1 {color: #d1a153; font-family: 'Georgia', serif;}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="amber", neutral_hue="slate"), css=custom_css) as demo:
    
    with gr.Div(elem_id="title-header"):
        gr.Markdown("# ⚙️ Karakuri Culinary Engine")
        gr.Markdown("### Autonomous Multi-Agent Gourmet Kitchen Dashboard & Wine Cellar Harmony")
    
    with gr.Row():
        with gr.Column(scale=4):
            # The structured chatbot element that renders HTML dialogue frames
            chatbot = gr.Chatbot(
                label="Kitchen Command Console", 
                type="messages",
                bubble_full_width=False,
                height=550
            )
            
            with gr.Row():
                txt_input = gr.Textbox(
                    show_label=False,
                    placeholder="Enter an ingredient, a conversion, or ask for a full gourmet menu concept...",
                    scale=8
                )
                submit_btn = gr.Button("Transmit Directive", variant="primary", scale=2)

        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ Engine Matrix")
            gr.DataFrame(
                value=[
                    ["Chef Module", "🧑‍🍳 Karakuri (Active)"],
                    ["Cellar Module", "🍷 Bacchus (Active)"],
                    ["Utility Core", "⚙️ Unit Conversions"],
                    ["Backbone LLM", f"Ollama: llama3.1" if USE_LOCAL else "OpenAI GPT-4o"]
                ],
                headers=["System", "Status"],
                interactive=False
            )
            gr.Markdown(
                "💡 **Example prompts to try:**\n"
                "- *'Design a gourmet roasted squab course.'*\n"
                "- *'Convert 62 Celsius to Fahrenheit for my water oven.'*\n"
                "- *'Help me fix a split béarnaise sauce.'*"
            )

    # Wire the interface event hooks
    submit_btn.click(run_culinary_session, inputs=[txt_input, chatbot], outputs=[txt_input, chatbot])
    txt_input.submit(run_culinary_session, inputs=[txt_input, chatbot], outputs=[txt_input, chatbot])

# ==========================================
# 6. RUN THE SYSTEM
# ==========================================
if __name__ == "__main__":
    # This fires up a local web server rendering full HTML pages automatically
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

