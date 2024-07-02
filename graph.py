from langgraph.graph import END, StateGraph

from graph_state import GraphState
from retriever import retrieveNODE
from retriever_grader import grade_documentsNODE
from generate import generateNODE

# create workflow graph
workFlow = StateGraph(GraphState)

# add nodes to graph
workFlow.add_node("retrieve", retrieveNODE)
workFlow.add_node("grade", grade_documentsNODE)
workFlow.add_node("generate", generateNODE)

# add edges between nodes, determines flow logic
workFlow.set_entry_point("retrieve")
workFlow.add_edge("retrieve", "grade")
workFlow.add_edge("grade", "generate")
workFlow.add_edge("generate", END)

print("Compilling graph...", end=" ", flush=True)
chatBot = workFlow.compile()
print("DONE")
