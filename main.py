from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Clase de estado (puedes agregar métodos propios si luego necesitas)
class State(dict): pass

# Modelo de lenguaje
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Nodo: lógica de decisión
def router(state):
    messages = state.get("messages", [])
    if not messages:
        return "ask_again"
    
    last_message = messages[-1].content.lower()
    if "resume" in last_message:
        return "summarize"
    elif "traduce" in last_message or "translate" in last_message:
        return "translate"
    elif "termina" in last_message or "end" in last_message:
        return END
    else:
        return "ask_again"

# Nodo: resumen
def summarize(state):
    content = state["messages"][-1].content
    prompt = HumanMessage(content=f"Resume esto en una oración: {content}")
    response = llm.invoke([prompt])
    state["messages"].append(response)
    return state

# Nodo: traducción
def translate(state):
    content = state["messages"][-1].content
    prompt = HumanMessage(content=f"Traduce esto al inglés: {content}")
    response = llm.invoke([prompt])
    state["messages"].append(response)
    return state

# Nodo: volver a preguntar
def ask_again(state):
    if "messages" not in state:
        state["messages"] = []
    prompt = HumanMessage(content="¿Qué deseas hacer ahora? (resume / traduce / termina)")
    state["messages"].append(prompt)
    return state

# Construcción del grafo
graph = StateGraph(State)

graph.add_node("ask_again", ask_again)
graph.add_node("router", router)
graph.add_node("summarize", summarize)
graph.add_node("translate", translate)

graph.set_entry_point("ask_again")
graph.add_edge("ask_again", "router")

graph.add_conditional_edges("router", router, {
    "summarize": "summarize",
    "translate": "translate",
    "ask_again": "ask_again",
    END: END
})

graph.add_edge("summarize", "ask_again")
graph.add_edge("translate", "ask_again")

# Compilar la app
app = graph.compile()

# Ejecutar la app con un estado inicial
if __name__ == "__main__":
    initial_state = State({
        "messages": [
            HumanMessage(content="Resume este párrafo: La inteligencia artificial es una rama de la informática que estudia cómo crear sistemas capaces de realizar tareas que requieren inteligencia humana.")
        ]
    })

    final_state = app.invoke(initial_state)

    print("\n📥 Conversación final:")
    for msg in final_state.get("messages", []):
        print("-", msg.content)
