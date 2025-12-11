from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

def pretty_print_state(state: dict):
    print("\n" + "="*80)
    print("🔵 STATE")
    print("="*80)
    for k, v in state.items():
        print(f"{k}: {v}")
    print("\n")

def pretty_print_system_prompt(prompt: str):
    print("="*80)
    print("🟣 SYSTEM PROMPT")
    print("="*80)
    print(prompt)
    print("\n")

def pretty_print_messages(response):
    print("="*80)
    print("🟠 ALL MESSAGES")
    print("="*80)

    msgs = response.get("messages", [])

    for idx, msg in enumerate(msgs):
        print(f"\n----- MESSAGE #{idx} -----")

        if isinstance(msg, HumanMessage):
            print("👤 HUMAN:")
            print(msg.content)

        elif isinstance(msg, AIMessage):
            print("🤖 AI:")
            print("Content:", msg.content)

            # Tool calls
            if msg.tool_calls:
                print("🔧 TOOL CALLS:")
                for tc in msg.tool_calls:
                    print(f"  Tool name: {tc['name']}")
                    print(f"  Args: {tc['args']}")
                    print(f"  Call ID: {tc['id']}")

        elif isinstance(msg, ToolMessage):
            print("🛠️ TOOL RESPONSE:")
            print("Raw content:", msg.content)

            if hasattr(msg, "artifact") and msg.artifact:
                print("\n📦 STRUCTURED CONTENT:")
                print(msg.artifact.get("structured_content"))

        else:
            print(f"(Unknown message type: {type(msg)}) content:")
            print(msg)
    
    print("\n")

def debug_all(state, system_prompt, response):
    pretty_print_state(state)
    pretty_print_system_prompt(system_prompt)
    pretty_print_messages(response)