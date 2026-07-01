import os
from dotenv import load_dotenv

# Load environment variables (API key, etc.)
load_dotenv()

# Import the LLM provider
from langchain_openai import ChatOpenAI

# Define the LLM instance
llm = ChatOpenAI(
    base_url="https://llm.brojs.ru/v1",
    api_key=os.getenv("BROJS_PAT_TOKEN"),
    model="openai/gpt-oss-20b",
    temperature=0.1,
)

# Import the tool that will be used by the agent
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Define a simple weather‑retrieval tool (placeholder implementation)
def get_weather(city: str, date: str) -> str:
    """Return a mock weather description for the given city and date."""
    return f"В {city} на {date} будет солнечно."

tools = [get_weather]

# Create an in‑memory checkpoint to preserve state across pauses
memory = MemorySaver()

# Build the agent using LangGraph’s prebuilt React agent
agent = create_react_agent(
    llm,
    tools,
    prompt="Ты полезный ассистент",
    checkpointer=memory,
)

def run_demo() -> str:
    """
    Run a single demo conversation with Human‑in‑the‑Loop pauses.
    The function demonstrates the pause/resume cycle and returns
    the final assistant message content.
    """
    # Initial user message
    config = {"configurable": {"thread_id": "сессия-1"}}
    result = agent.invoke(
        {"messages": [{"role": "human", "content": "Какая погода в Казани сегодня?"}]},
        config=config,
    )

    # Loop until the agent finishes (no more pauses)
    while "__interrupt__" in result:
        interrupt_value = result["__interrupt__"][0].value
        action_requests = interrupt_value["action_requests"]
        review_configs = interrupt_value["review_configs"]

        decisions = []
        for idx, action in enumerate(action_requests):
            name = action.get("name")
            args = action.get("args", {})
            description = action.get("description", "")

            # Display the request to the user
            print("\n--- Подтверждение ---")
            print(f"Инструмент: {name}")
            print(f"Аргументы: {args}")
            if description:
                print(f"Описание: {description}")

            # Ask for a decision (approve/reject)
            while True:
                choice = input("a = approve, r = reject: ").strip().lower()
                if choice == "a":
                    decisions.append({"type": "approve"})
                    break
                elif choice == "r":
                    msg = input(
                        "Сообщение для агента (причина отказа): "
                    ).strip()
                    decisions.append({"type": "reject", "message": msg})
                    break
                else:
                    print("Неверный ввод. Пожалуйста, введите 'a' или 'r'.")

        # Resume the agent with the collected decisions
        result = agent.invoke(
            {"resume": {"decisions": decisions}},
            config=config,
        )

    # After exiting the loop, extract final assistant message
    final_message = result["messages"][-1].content
    return final_message

if __name__ == "__main__":
    demo_result = run_demo()
    print("\n--- Финальный ответ ---")
    print(demo_result)
