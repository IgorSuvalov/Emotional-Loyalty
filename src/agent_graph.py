import json
from typing import TypedDict, Any, Optional, Literal
from langgraph.graph import StateGraph, END
from src.llm import llm
from src.agent_tools import run_model, ModelParams, _sanitize


Intent = Literal["EXPLAIN_MODEL", "SUGGEST_AND_RUN"]


class AgentState(TypedDict):
    user_goal: str
    intent: Intent
    params: ModelParams
    last_summary: Optional[dict]
    df: Any
    response: str


def _safe_json(s: Any) -> dict:
    if s is None:
        return {}
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if not s:
        return {}

    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1 :]
        if s.endswith("```"):
            s = s[: -3]
        s = s.strip()

    try:
        return json.loads(s)
    except Exception:
        return {}


def _default_params():
    return ModelParams(
        spend=0.6,
        engage=0.4,
        lam=0.8,
        multipliers={
            'Brand Champions': 1.05,
            'Transactional Spenders': 0.95,
            'Brand Advocates': 1.10,
            'Passive Customers': 0.90
        },
        tier_mix={
            'Platinum': 0.10,
            'Gold': 0.15,
            'Silver': 0.25,
            'Regular': 0.50
        }
    )


def classify_intent(state: AgentState):
    prompt = (
        "Classify the user's goal for a loyalty-scoring model into one of two categories."
        "Return JSON: {\"intent\": <INTENT>} where <INTENT> is either \"EXPLAIN_MODEL\" or \"SUGGEST_AND_RUN\"."
    )
    output = llm.invoke(f"{prompt}\nUser goal: {state['user_goal']}\nJSON:").content
    data = _safe_json(output)
    text = state["user_goal"].lower()
    state["intent"] = data.get("intent") or (
        "EXPLAIN_MODEL" if ("explain" in text and "param" not in text) else "SUGGEST_AND_RUN"
    )
    return state


def explain_model(state: AgentState):
    explanation = (
        "Explain for the loyalty scoring model works for a business audience in 5-6 sentences."
        "Cover the spend and engage blocks, U(x) = weighted sum of two blocks, where weights are knobs."
        "Cover the archetypes and the unsupervised signal."
        "Cover the lambda parameter for the confidence between supervised and unsupervised signal."
        "Note that business' can choose to boost or penalise the scores (+-20%) of customers based on their archetype, "
        "and how tiers are distributed. Avoid jargon. Be concrete."
    )

    state["response"] = llm.invoke(f"{explanation}\n").content
    return state


def suggest_and_run(state: AgentState):
    default_params = _default_params()
    prompt = (
        "Based on the user's goal, suggest values for the loyalty scoring model parameters."
        "Return JSON with keys: spend (float 0-1), engage (float 0-1), lam (float 0-1),"
        "multipliers (dict of archetype to float), tier_mix (dict of tier to float)."
        "Ensure spend + engage = 1.0. Make sure tier_mix sums to 1.0. Those two constraints are MANDATORY."
        "Unless otherwise specified keep tier_mix as in the default"
        "Below you will see the default parameters for reference. Alter then according to the user's goal."
    )
    default_json = json.dumps({
        "spend": default_params.spend,
        "engage": default_params.engage,
        "lam": default_params.lam,
        "multipliers": default_params.multipliers,
        "tier_mix": default_params.tier_mix
    })
    output = llm.invoke(f"{prompt}\nUser goal: {state['user_goal']}\nDefault JSON for reference{default_json}\nJSON:").content
    data = _safe_json(output)

    params = _sanitize(data, default_params)

    final_df, _ = run_model(state["df"], params)

    state["params"] = params
    state["response"] = (
        f"Model run complete. Here is the summary of results:\n"
    )
    return state

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("classify_intent", classify_intent)
    g.add_node("explain_model", explain_model)
    g.add_node("suggest_and_run", suggest_and_run)

    g.set_entry_point("classify_intent")

    def route(state: AgentState):
        if state["intent"] == "EXPLAIN_MODEL":
            return "explain_model"
        else:
            return "suggest_and_run"

    g.add_conditional_edges("classify_intent", route, {
        "explain_model": "explain_model",
        "suggest_and_run": "suggest_and_run"
    })

    g.add_edge("explain_model", END)
    g.add_edge("suggest_and_run", END)

    return g.compile()

