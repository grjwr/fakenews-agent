from typing import TypedDict, List, Optional, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    claim: str
    messages: Annotated[list, add_messages]
    evidence: List[dict]
    eprvfl_result: Optional[dict]
    mistral_result: Optional[dict]
    verdict: Optional[dict]
    route: str
