from graph_state import GraphState
from graph import chatBot

QUESTION = "Who is Oliver"

state = GraphState()
state["question"] = QUESTION
result = chatBot.invoke(state)

print(result["answer"])
