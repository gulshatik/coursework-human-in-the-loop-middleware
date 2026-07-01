# Human-in-the-Loop через middleware

## Что такое Human‑in‑the‑Loop (HITL)?

Human‑in‑the‑Loop – это подход, при котором человек участвует в процессе принятия решений модели.  
В контексте LangChain/LangGraph это реализуется через **middleware**, который останавливает агента перед выполнением инструмента и запрашивает у пользователя подтверждение (`approve`), отклонение (`reject`) или редактирование аргументов (`edit`).  

## Задача

Нужно создать агент, использующий `HumanInTheLoopMiddleware`, чтобы при каждом вызове инструмента пользователь видел запрос на подтверждение и мог решить:

- **Approve** – продолжить выполнение.
- **Reject** – отменить действие (можно добавить сообщение‑обратную связь).
- **Edit** – изменить аргументы перед выполнением.  
  *В примере реализованы только `approve` и `reject`, но структура готова к добавлению `edit`.*

## Структура проекта

```
.
├── main.py          # основной скрипт с циклом HITL
├── requirements.txt # зависимости
└── README.md        # данная документация
```

---

## 1. Установка зависимостей

```bash
# Создайте и активируйте виртуальное окружение (по желанию)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Установите зависимости из requirements.txt
pip install -r requirements.txt
```

**requirements.txt**

```text
langchain==0.2.*
langgraph==0.1.*
openai>=1.0
```

> **Важно:** версии указаны как шаблоны (`*`). Вы можете заменить их на актуальные при установке.

---

## 2. Определение инструмента

В примере используется простая функция `get_weather`, которая возвращает заглушку погоды.  
Вы можете подключить реальный API, если захотите.

```python
# main.py (часть)
from langchain.tools import Tool, tool

@tool("get_weather", "Возвращает погоду для заданного города и даты.")
def get_weather(city: str, date: str) -> str:
    # Заглушка – замените реальным API при необходимости
    return f"Погода в {city} на {date}: солнечно."
```

---

## 3. Создание LLM

```python
# main.py (часть)
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)   # можно задать свой API‑ключ через переменные окружения
```

> **Tip:** Если вы используете OpenAI, убедитесь, что переменная `OPENAI_API_KEY` установлена в системе.

---

## 4. Инициализация памяти и агента

```python
# main.py (часть)
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

memory = MemorySaver()

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="Ты полезный ассистент",
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"get_weather": True},  # все решения: approve, edit, reject
            description_prefix="Подтвердите вызов инструмента",
        ),
    ],
    checkpointer=memory,
)
```

> **Почему нужен `MemorySaver`?**  
> Без него агент не сможет сохранить состояние паузы и продолжить работу после подтверждения.

---

## 5. Функция цикла HITL

```python
# main.py (часть)
def run_agent_with_hil(user_msg: str, thread_id: str):
    """
    Запускает агента с Human‑in‑the‑Loop.
    user_msg   – сообщение пользователя
    thread_id  – уникальный идентификатор сессии (для сохранения истории)
    """
    config = {"configurable": {"thread_id": thread_id}}

    # Первый вызов агента
    result = agent.invoke(
        {"messages": [{"role": "human", "content": user_msg}]},
        config=config,
    )

    # Цикл: пока агент запрашивает подтверждение
    while "__interrupt__" in result:
        interrupt_value = result["__interrupt__"][0].value
        action_requests = interrupt_value["action_requests"]
        decisions = []

        for idx, act in enumerate(action_requests):
            print(f"\n--- Подтверждение #{idx + 1} ---")
            print(f"Инструмент: {act['name']}")
            print(f"Аргументы: {act.get('args')}")
            if "description" in act:
                print(f"Описание: {act['description']}")

            # Запрос решения от пользователя
            while True:
                choice = input("a=approve, r=reject (e=edit): ").strip().lower()
                if choice == "a":
                    decisions.append({"type": "approve"})
                    break
                elif choice == "r":
                    msg = input("Сообщение для агента: ")
                    decisions.append({"type": "reject", "message": msg})
                    break
                elif choice == "e":
                    # Простая реализация edit (необязательно)
                    new_args = input(
                        "Введите отредактированные аргументы в формате JSON: "
                    )
                    try:
                        import json

                        edited = json.loads(new_args)
                        decisions.append(
                            {
                                "type": "edit",
                                "edited_action": {"name": act["name"], "args": edited},
                            }
                        )
                        break
                    except Exception as e:
                        print(f"Ошибка парсинга JSON: {e}")
                else:
                    print("Неверный ввод, попробуйте снова.")

        # Возобновляем выполнение агента с решениями
        result = agent.invoke(Command(resume={"decisions": decisions}), config=config)

    # После завершения выводим финальный ответ
    final_msg = result["messages"][-1]["content"]
    print("\nАгент:", final_msg)
```

---

## 6. Главная часть скрипта

```python
# main.py (часть)
if __name__ == "__main__":
    thread_id = "session-1"   # можно менять для разных сессий

    while True:
        user_input = input("\nВы: ")
        if user_input.lower() in {"exit", "quit"}:
            print("До свидания!")
            break
        run_agent_with_hil(user_input, thread_id)
```

---

## 7. Запуск

```bash
python main.py
```

### Пример работы

```
Вы: Какая погода в Казани сегодня?

--- Подтверждение #1 ---
Инструмент: get_weather
Аргументы: {'city': 'Казань', 'date': 'сегодня'}
a=approve, r=reject (e=edit): a

Агент: Погода в Казань на сегодня: солнечно.
```

При отклонении:

```
Вы: Какая погода в Казани сегодня?

--- Подтверждение #1 ---
Инструмент: get_weather
Аргументы: {'city': 'Казань', 'date': 'сегодня'}
a=approve, r=reject (e=edit): r
Сообщение для агента: Не нужно

Агент: Похоже, вы не хотите знать погоду. Чем ещё могу помочь?
```

---

## 8. Что дальше?

- **Редактирование аргументов** – реализуйте более удобный ввод JSON или форму по полям.
- **Множественные инструменты** – добавьте несколько инструментов и настройте `interrupt_on` для каждого.
- **Пользовательский интерфейс** – вместо консоли можно использовать веб‑UI (Streamlit, Gradio) с кнопками подтверждения/отказа.

---

## 9. Полный код (`main.py`)

```python
from langchain.tools import Tool, tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

# 1. Инструмент
@tool("get_weather", "Возвращает погоду для заданного города и даты.")
def get_weather(city: str, date: str) -> str:
    return f"Погода в {city} на {date}: солнечно."

# 2. LLM
llm = ChatOpenAI(temperature=0)

# 3. Память + агент
memory = MemorySaver()
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="Ты полезный ассистент",
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"get_weather": True},
            description_prefix="Подтвердите вызов инструмента",
        )
    ],
    checkpointer=memory,
)

# 4. Цикл HITL
def run_agent_with_hil(user_msg: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "human", "content": user_msg}]},
        config=config,
    )
    while "__interrupt__" in result:
        interrupt_value = result["__interrupt__"][0].value
        action_requests = interrupt_value["action_requests"]
        decisions = []
        for idx, act in enumerate(action_requests):
            print(f"\n--- Подтверждение #{idx + 1} ---")
            print(f"Инструмент: {act['name']}")
            print(f"Аргументы: {act.get('args')}")
            if "description" in act:
                print(f"Описание: {act['description']}")
            while True:
                choice = input("a=approve, r=reject (e=edit): ").strip().lower()
                if choice == "a":
                    decisions.append({"type": "approve"})
                    break
                elif choice == "r":
                    msg = input("Сообщение для агента: ")
                    decisions.append({"type": "reject", "message": msg})
                    break
                elif choice == "e":
                    new_args = input(
                        "Введите отредактированные аргументы в формате JSON: "
                    )
                    try:
                        import json

                        edited = json.loads(new_args)
                        decisions.append(
                            {
                                "type": "edit",
                                "edited_action": {"name": act["name"], "args": edited},
                            }
                        )
                        break
                    except Exception as e:
                        print(f"Ошибка парсинга JSON: {e}")
                else:
                    print("Неверный ввод, попробуйте снова.")
        result = agent.invoke(Command(resume={"decisions": decisions}), config=config)
    final_msg = result["messages"][-1]["content"]
    print("\nАгент:", final_msg)

# 5. Главная часть
if __name__ == "__main__":
    thread_id = "session-1"
    while True:
        user_input = input("\nВы: ")
        if user_input.lower() in {"exit", "quit"}:
            print("До свидания!")
            break
        run_agent_with_hil(user_input, thread_id)
```

---

## 10. Заключение

С помощью `HumanInTheLoopMiddleware` вы получаете гибкий механизм взаимодействия пользователя с агентом без необходимости писать собственный код для паузы и подтверждения.  
Весь процесс сохраняется в памяти, поэтому можно продолжать диалог в рамках одной сессии, а также легко масштабировать под более сложные сценарии.

---
