from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graph_state import GraphState
from ollama_model import llm

# it is very important to choose prompt in language of expected question for optimal performance
promptEN = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'YES' or 'NO' score to indicate whether the document is relevant to the question without preamble or explanation.""",
    input_variables=["question", "document"],
)

retrieval_grader = promptEN | llm | StrOutputParser()

def grade_document(question, document):
    answer = retrieval_grader.invoke({"question": question, "document": document})
    if answer == "YES":
        return True
    else:
        return False
    
# function for langgraph
def grade_documentsNODE(state : GraphState):
    question = state["question"]
    documents = state["documents"]
    filtered = []
    for doc in documents:
        print("Grading document...", end=" ", flush=True)
        answer = retrieval_grader.invoke({"question": question, "document": doc})
        if answer == "YES":
            filtered.append(doc)
            print("RELEVANT")
        else:
            print("IRRELEVANT")
    return {"documents": filtered}


# for testing purposes
if __name__ == "__main__":
    document = 'One rainy afternoon, Oliver returned to the bookstore to thank Agatha. He found her sitting by the fireplace, a cup of tea in hand and a knowing smile on her lips. She had seen the transformation in him, the spark reignited in his eyes.\n\n"Thank you," Oliver said sincerely, "for helping me find my voice again."'
    question = "Who is oliver"
    print(retrieval_grader.invoke({"question": question, "document": document}))
