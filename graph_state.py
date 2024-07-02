from typing import List
from typing_extensions import TypedDict

class GraphState(TypedDict):
    question: str
    answer: str
    documents: List[str]