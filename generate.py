from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graph_state import GraphState
from ollama_model import llm

# it is very important to choose prompt in language of expected question for optimal performance
promptEN = PromptTemplate(
    template="You are an assistant for question-answering tasks. \
              Use the following pieces of retrieved context to answer the question. \
              If you don't know the answer, just say 'Unfortunately I could not find an answer, please contact us at michal@gift.cz'. \
              Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:",
    input_variables=['context', 'question'],
)

# do not forget to set correct prompt lang
chat = promptEN | llm | StrOutputParser()

# function for langgraph
def generateNODE(state : GraphState):
    print("Generating answer...", end=" ", flush=True)
    context = state["documents"]
    question = state["question"]
    answer = chat.invoke({"context": context, "question": question})
    print("DONE")
    return {"answer": answer}
