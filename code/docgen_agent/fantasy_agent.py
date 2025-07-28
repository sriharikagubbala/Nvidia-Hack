"""
The main agent that analyzes player trends and recommends Fantasy Football draft picks.
"""

import asyncio
import logging
import os
from typing import Annotated, Any, Sequence, cast

from langchain_core.runnables import RunnableConfig
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from . import author, researcher  # You can later change or add your fantasy tools here
from .prompts import fantasy_prompt_intro  # Make sure to define this in prompts.py

_LOGGER = logging.getLogger(__name__)
_MAX_LLM_RETRIES = 3
_QUERIES_PER_SECTION = 5
_THROTTLE_LLM_CALLS = os.getenv("THROTTLE_LLM_CALLS", "0")

llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0)


class DraftPlan(BaseModel):
    title: str
    recommendations: list[author.Section]  # You can rename Section later if needed


class FantasyAgentState(BaseModel):
    topic: str
    league_type: str
    draft_plan: DraftPlan | None = None
    recommendation_output: str | None = None
    messages: Annotated[Sequence[Any], add_messages] = []


async def player_research(state: FantasyAgentState, config: RunnableConfig):
    """Research trending players and relevant stats."""
    _LOGGER.info("Performing player research.")

    researcher_state = researcher.ResearcherState(
        topic=state.topic,
        number_of_queries=_QUERIES_PER_SECTION,
        messages=state.messages,
    )

    research = await researcher.graph.ainvoke(researcher_state, config)
    return {"messages": research.get("messages", [])}


async def draft_planner(state: FantasyAgentState, config: RunnableConfig):
    """Call LLM to generate draft strategy."""
    _LOGGER.info("Calling draft planner.")

    model = llm.with_structured_output(DraftPlan)  # type: ignore

    system_prompt = fantasy_prompt_intro.format(
        topic=state.topic,
        league_type=state.league_type,
    )

    for count in range(_MAX_LLM_RETRIES):
        messages = [{"role": "system", "content": system_prompt}] + list(state.messages)
        response = await model.ainvoke(messages, config)
        if response:
            response = cast(DraftPlan, response)
            state.draft_plan = response
            return state
        _LOGGER.debug("Retrying LLM call. Attempt %d of %d", count + 1, _MAX_LLM_RETRIES)

    raise RuntimeError("Failed to call model after %d attempts.", _MAX_LLM_RETRIES)


async def player_analysis_orchestrator(state: FantasyAgentState, config: RunnableConfig):
    """Coordinate analysis of each recommended player."""
    if not state.draft_plan:
        raise ValueError("Draft plan is not set.")

    _LOGGER.info("Analyzing each player in the draft plan.")

    writers = []
    for idx, rec in enumerate(state.draft_plan.recommendations):
        _LOGGER.info("Creating analysis for: %s", rec.name)

        player_analysis_state = author.SectionWriterState(
            index=idx,
            section=rec,
            topic=state.topic,
            messages=state.messages,
        )
        writers.append(author.graph.ainvoke(player_analysis_state, config))

    all_analyses = []
    if _THROTTLE_LLM_CALLS == "1":
        _LOGGER.info("Throttling LLM calls.")
        for writer in writers:
            all_analyses.append(await writer)
            await asyncio.sleep(30)
    else:
        all_analyses = await asyncio.gather(*writers)
    all_analyses = cast(list[dict[str, Any]], all_analyses)

    for analysis in all_analyses:
        index = analysis["index"]
        content = analysis["section"].content
        state.draft_plan.recommendations[index].content = content
        _LOGGER.info("Finished analysis for: %s", state.draft_plan.recommendations[index].name)

    return state


async def draft_summary_writer(state: FantasyAgentState, config: RunnableConfig):
    """Compile final draft recommendation output."""
    if not state.draft_plan:
        raise ValueError("Draft plan is not set.")

    _LOGGER.info("Writing final draft recommendation.")

    output = f"# {state.draft_plan.title}\n\n"
    for rec in state.draft_plan.recommendations:
        output += rec.content
        output += "\n\n"

    state.recommendation_output = output
    return state


# Build the graph
workflow = StateGraph(FantasyAgentState)

workflow.add_node("player_research", player_research)
workflow.add_node("draft_planner", draft_planner)
workflow.add_node("player_analysis_orchestrator", player_analysis_orchestrator)
workflow.add_node("draft_summary_writer", draft_summary_writer)

workflow.add_edge(START, "player_research")
workflow.add_edge("player_research", "draft_planner")
workflow.add_edge("draft_planner", "player_analysis_orchestrator")
workflow.add_edge("player_analysis_orchestrator", "draft_summary_writer")
workflow.add_edge("draft_summary_writer", END)

graph = workflow.compile()
