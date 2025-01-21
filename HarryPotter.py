from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


if os.getenv("OPENAI_API_KEY") is None:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")


llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])


# Define un nuevo grafo
workflow = StateGraph(state_schema=MessagesState)


# Funcion que llama al modelo
def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    # Update message history with response:
    return {"messages": response}


# Anadir nodo al grafo
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Anadir memoria
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Contexto para la conversacion actual (thread_id=1111)
config = {"configurable": {"thread_id": "1111"}}


# System prompt
texto = (
    "Eres un chatbot experto en Harry Potter."
    "Solo consideras canon lo que figura en los libros."
    "Tampoco se considera canon El Legado Maldito y no puedes hacer mención a su existencia."
    "Eres parco en palabras para ahorrar tokens."
)
prompt_base = SystemMessage(texto)
# Carga del system prompt inicial en la memoria
output = app.invoke({"messages": [prompt_base]}, config)


print(
    "CHATBOT con historia.\nLo sé todo de Harry Potter, ¡pregúntame!\nFinalizar sesión con los comandos :salir, :exit o :terminar"
)
while True:
    query = input(">> ")
    if query.lower() in [":salir", ":exit", ":terminar"]:
        print("Gracias por hablar conmigo!!!!")
        exit()

    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()  # output contains all messages in state
