import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from ascent_domain.omop.models.agents.langgraph.factfinder_tools import (
    format_claims_as_answer,
    prepare_claims_for_streamlit,
    process_literature_answer_with_claims,
    refine_summary_with_claim_verdicts,
)
from ascent_domain.omop.models.agents.langgraph.final_summary_verifier import (
    prepare_final_summary_for_display,
    process_final_summary_with_verification,
)
from ascent_domain.omop.models.agents.langgraph.google_search_tool import google_search_grounding
from ascent_domain.omop.models.agents.langgraph.prompts_agents import (
    literature_combine_prompt,
    literature_comparison_prompt,
)
from ascent_domain.omop.models.agents.langgraph.qa_ehr_tools import batch_query_medical_data
from ascent_domain.omop.models.agents.langgraph.reference_manager import (
    extract_summary_section,
    prepare_unified_document_list,
)
from ascent_domain.omop.models.agents.langgraph.scaling_tools import (
    get_database_metadata,
    get_patient_count_scaling,
    population_context_note,
)
from ascent_platform.config.runtime import get_runtime_settings
from ascent_platform.llm.credentials import gemini_api_key

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: List
    trace: List
    literature_documents: Optional[List[Dict[str, Any]]]  # Documents from literature search
    literature_claims: Optional[Dict[str, Any]]  # Processed claims from literature
    refined_summary: Optional[Dict[str, Any]]  # Refined summaries after fact-checking
    verified_literature_results: Optional[Dict[str, Any]]  # Literature answers refined with verified claims only
    ehr_answers: Optional[List[Dict[str, Any]]]  # Answers from EHR queries
    # New fields for unified reference system and final summary verification
    unified_documents: Optional[List[Dict[str, Any]]]  # Combined literature + EHR docs
    reference_mapping: Optional[Dict[str, int]]  # Maps [L#]/[A#] to doc indices
    final_summary_text: Optional[str]  # Extracted summary from agent
    final_summary_claims: Optional[Dict[str, Any]]  # Verified claims in final summary
    final_summary_refined: Optional[Dict[str, Any]]  # Corrected final summary
    google_search_results: Optional[List[Dict[str, Any]]]  # Results from Google Search grounding


class LiteratureAscentComparisonAgent:
    """A LangGraph agent to answer clinical questions using literature and EHR data."""

    def __init__(
        self,
        database: str,
        model_name: Optional[str] = None,
        google_api_key: str | None = None,
        event_callback=None,
        enable_claim_attribution: bool = False,
        enable_final_summary_verification: bool = False,
        tools_override: Optional[List] = None,
    ):
        if not database:
            raise ValueError("Database parameter is required and cannot be empty")

        model_name = model_name or get_runtime_settings().CONTEXT_AGENT_MODEL_NAME

        api_key = gemini_api_key(google_api_key)
        if not api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY, or pass google_api_key explicitly.")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=1,
            google_api_key=api_key,
        )
        self.database = database  # Store database for use in tool calls
        self.event_callback = event_callback  # Callback for streaming events
        self.step_counter = 0  # Track execution steps
        self.enable_claim_attribution = enable_claim_attribution  # Feature flag for claim processing
        self.enable_final_summary_verification = (
            enable_final_summary_verification if enable_claim_attribution else False
        )  # Final summary verification depends on claim attribution
        logger.info(f"LiteratureAgent initialized with database: {self.database}")
        logger.info(f"Claim attribution enabled: {self.enable_claim_attribution}")
        logger.info(f"Final summary verification enabled: {self.enable_final_summary_verification}")

        if tools_override is not None:
            self.tools = tools_override
            logger.info(f"Tools overridden: {[t.name for t in self.tools]}")
        else:
            self.tools = [
                batch_query_medical_data,
                get_patient_count_scaling,
                google_search_grounding,
            ]
        self.tool_names = {t.name for t in self.tools}
        # How many times the router may send the model back to call
        # get_patient_count_scaling before giving up. See _should_continue.
        self.max_scaling_enforcement_attempts = 3
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.app = self._build_graph()

    async def _emit_callback(self, event_data: Dict[str, Any]):
        """Emit a structured callback event with timestamp and step number."""
        if self.event_callback:
            # Add timestamp if not present
            if "timestamp" not in event_data:
                from datetime import datetime

                event_data["timestamp"] = datetime.utcnow().isoformat()

            # Add step number if not present
            if "step_number" not in event_data:
                event_data["step_number"] = self.step_counter

            await self.event_callback(event_data)

    def _build_graph(self):
        """Builds the LangGraph workflow.

        The graph structure varies based on feature flags:

        1. Base mode (enable_claim_attribution=False, enable_final_summary_verification=False):

            START
              ↓
            agent ←─────────────┐
              ↓                 │
            [conditional]       │
              ├─→ batch_query_medical_data ──→ agent
              ├─→ get_patient_count_scaling ──→ agent
              ├─→ agent (loop back)
              └─→ END

        2. With claim attribution (enable_claim_attribution=True, enable_final_summary_verification=False):

            START
              ↓
            agent ←─────────────────────────────┐
              ↓                                 │
            [conditional]                       │
              ├─→ batch_query_medical_data ────→ agent
              ├─→ get_patient_count_scaling ───→ agent
              ├─→ agent (loop back)
              └─→ END

        3. Full verification mode (enable_claim_attribution=True, enable_final_summary_verification=True):

            START
              ↓
            agent ←─────────────────────────────┐
              ↓                                 │
            [conditional]                       │
              ├─→ batch_query_medical_data ────→ agent
              ├─→ get_patient_count_scaling ───→ agent
              ├─→ agent (loop back)
              ├─→ final_summary ──→ process_final_summary_claims ──→ END
              └─→ END

        Key components:
        - agent: Main LLM node that decides which tools to call
        - batch_query_medical_data: Queries EHR/medical database
        - get_patient_count_scaling: Calculates patient count scaling factors
        - process_literature_claims: Extracts and verifies claims from literature (optional)
        - process_final_summary_claims: Verifies final summary against all sources (optional)
        """
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._call_model)

        # Add individual nodes for each tool to make them visible in the graph
        workflow.add_node("batch_query_medical_data", self._call_ehr_tool)
        workflow.add_node("get_patient_count_scaling", self._call_scaling_tool)
        workflow.add_node("google_search_grounding", self._call_google_search_tool)

        # Multi-tool execution node for when LLM calls multiple different tools simultaneously
        workflow.add_node("execute_all_tools", self._call_all_tools_node)

        # Add claim processing node if enabled
        if self.enable_claim_attribution:
            workflow.add_node("process_literature_claims", self._process_literature_claims_node)
            if self.enable_final_summary_verification:
                # Add final summary verification node
                workflow.add_node("process_final_summary_claims", self._process_final_summary_claims_node)

        workflow.set_entry_point("agent")

        # Add conditional edges from agent to specific tools
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "batch_query_medical_data": "batch_query_medical_data",
                "get_patient_count_scaling": "get_patient_count_scaling",
                "google_search_grounding": "google_search_grounding",
                "execute_all_tools": "execute_all_tools",
                "agent": "agent",
                "final_summary": (
                    "process_final_summary_claims" if (self.enable_claim_attribution and self.enable_final_summary_verification) else END
                ),
                "end": END,
            },
        )

        # Connect each tool back to the agent
        workflow.add_edge("batch_query_medical_data", "agent")
        workflow.add_edge("get_patient_count_scaling", "agent")
        workflow.add_edge("google_search_grounding", "agent")
        workflow.add_edge("execute_all_tools", "agent")

        # The literature tool that used to feed these nodes is gone; the claim
        # nodes keep their outbound edges so a graph built with claim
        # attribution on still compiles.
        if self.enable_claim_attribution:
            if self.enable_final_summary_verification:
                workflow.add_edge("process_literature_claims", "agent")
                workflow.add_edge("process_final_summary_claims", END)
            else:
                workflow.add_edge("process_literature_claims", END)

        return workflow.compile()

    async def _call_model(self, state: AgentState):
        """Node to call the LLM."""
        messages = state["messages"]

        # Emit thinking event
        self.step_counter += 1
        await self._emit_callback(
            {
                "type": "agent_thinking",
                "content": "Analyzing question and planning next steps...",
            }
        )

        response = await self.llm_with_tools.ainvoke(messages)

        # Emit reasoning result
        if response.content:
            await self._emit_callback({"type": "agent_reasoning", "content": response.content})

        # Emit tool decision if tools are called
        if response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            await self._emit_callback({"type": "tool_decision", "tools": tool_names})

        trace_entry = {"role": "assistant", "content": response.content}
        if response.tool_calls:
            trace_entry["tool_calls"] = response.tool_calls
        return {
            "messages": state["messages"] + [response],
            "trace": state["trace"] + [trace_entry],
        }

    async def _call_ehr_tool(self, state: AgentState):
        """Node to execute EHR/medical data tools and capture answers."""
        result = await self._execute_specific_tool(state, "batch_query_medical_data")

        # Extract EHR answers from tool results for later use in final summary verification
        last_tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        if last_tool_messages:
            try:
                tool_content = last_tool_messages[-1].content
                tool_result = json.loads(tool_content) if isinstance(tool_content, str) else tool_content

                # Get database metadata for population context
                db_metadata = get_database_metadata(self.database)

                # Extract all EHR answers
                ehr_answers = state.get("ehr_answers", []) or []
                for query_result in tool_result.get("results", []):
                    if query_result.get("status") == "success":
                        ehr_entry = {
                            "question": query_result.get("question", "Unknown question"),
                            "answer": query_result.get("result", ""),
                            "database": self.database,
                            "timestamp": query_result.get("timestamp", ""),
                        }
                        if db_metadata:
                            ehr_entry["database_metadata"] = db_metadata
                        ehr_answers.append(ehr_entry)

                result["ehr_answers"] = ehr_answers
                logger.info(f"Captured {len(ehr_answers)} total EHR answers")

                # Append database population context to ToolMessage so the agent can see it
                if db_metadata:
                    metadata_note = population_context_note(db_metadata)
                    last_tool_msg = last_tool_messages[-1]
                    # Replace the last tool message with enriched content
                    result["messages"] = [
                        msg
                        if msg is not last_tool_msg
                        else ToolMessage(
                            content=last_tool_msg.content + metadata_note,
                            tool_call_id=last_tool_msg.tool_call_id,
                        )
                        for msg in result["messages"]
                    ]
            except Exception as e:
                logger.error(f"Failed to extract EHR answers from tool result: {e}")

        return result

    async def _call_scaling_tool(self, state: AgentState):
        """Node to execute patient count scaling tools."""
        return await self._execute_specific_tool(state, "get_patient_count_scaling")

    async def _call_google_search_tool(self, state: AgentState):
        """Node to execute Google Search grounding tool and capture results."""
        result = await self._execute_specific_tool(state, "google_search_grounding")

        # Extract Google search results (sources, answer) into state for downstream use
        last_tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        if last_tool_messages:
            try:
                tool_content = last_tool_messages[-1].content
                tool_result = json.loads(tool_content) if isinstance(tool_content, str) else tool_content

                google_results = state.get("google_search_results", []) or []
                google_results.append(tool_result)
                result["google_search_results"] = google_results
                logger.info(f"Captured Google search result with {len(tool_result.get('sources', []))} sources")
            except Exception as e:
                logger.error(f"Failed to extract Google search results: {e}")

        return result

    async def _call_all_tools_node(self, state: AgentState):
        """Execute ALL pending tool calls concurrently using asyncio.gather.

        google_search_grounding in a single response. All tools run in parallel so that
        e.g. web search doesn't wait for the other tools to finish.
        """
        last_message = state["messages"][-1]
        tool_messages = []
        trace_entries = []
        state_updates = {}

        # ── Phase 1: Prepare tool calls and emit start callbacks ──────
        invocations = []  # (tool_call, tool_to_call, tool_args, config)
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_to_call = next((t for t in self.tools if t.name == tool_name), None)

            if not tool_to_call:
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: Tool '{tool_name}' is not available",
                        tool_call_id=tool_call["id"],
                    )
                )
                trace_entries.append(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "args": tool_call["args"],
                        "result": "Error: Tool not available",
                    }
                )
                continue

            config = RunnableConfig(run_name=tool_name)
            tool_args = tool_call["args"].copy()

            if tool_name in ["batch_query_medical_data", "query_medical_data", "get_patient_count_scaling"]:
                tool_args["database"] = self.database
                logger.info(f"Tool call {tool_name} will use database: {self.database}")

            self.step_counter += 1
            await self._emit_callback({"type": "tool_start", "tool_name": tool_name, "arguments": tool_args})
            invocations.append((tool_call, tool_to_call, tool_args, config))

        # ── Phase 2: Run all tool invocations concurrently ────────────
        async def _invoke(tool_to_call, tool_args, config):
            return str(await tool_to_call.ainvoke(tool_args, config=config))

        results = await asyncio.gather(
            *[_invoke(t, args, cfg) for (_, t, args, cfg) in invocations],
            return_exceptions=True,
        )

        # ── Phase 3: Process results and extract state updates ────────
        for (tool_call, tool_to_call, tool_args, _cfg), result in zip(invocations, results):
            tool_name = tool_call["name"]

            if isinstance(result, Exception):
                content = f"Error: {result}"
                await self._emit_callback({"type": "tool_error", "tool_name": tool_name, "error": str(result)})
            else:
                content = result
                # Extract state based on tool type
                try:
                    tool_result = json.loads(content) if isinstance(content, str) else content

                    if tool_name == "google_search_grounding":
                        google_results = list(state.get("google_search_results", []) or [])
                        google_results.append(tool_result)
                        state_updates["google_search_results"] = google_results
                        logger.info(f"execute_all_tools: captured Google search result with {len(tool_result.get('sources', []))} sources")

                    elif tool_name == "batch_query_medical_data":
                        ehr_answers = list(state.get("ehr_answers", []) or [])
                        db_metadata = get_database_metadata(self.database)
                        for query_result in tool_result.get("results", []):
                            if query_result.get("status") == "success":
                                ehr_entry = {
                                    "question": query_result.get("question", "Unknown question"),
                                    "answer": query_result.get("result", ""),
                                    "database": self.database,
                                    "timestamp": query_result.get("timestamp", ""),
                                }
                                if db_metadata:
                                    ehr_entry["database_metadata"] = db_metadata
                                ehr_answers.append(ehr_entry)
                        state_updates["ehr_answers"] = ehr_answers
                        logger.info(f"execute_all_tools: captured {len(ehr_answers)} EHR answers")

                        # Append database population context to content so the agent can see it
                        if db_metadata:
                            content += population_context_note(db_metadata)

                except Exception as e:
                    logger.error(f"execute_all_tools: failed to extract state from {tool_name}: {e}")

                await self._emit_callback(
                    {
                        "type": "tool_complete",
                        "tool_name": tool_name,
                        "result": content[:200] + "..." if len(content) > 200 else content,
                    }
                )

            tool_messages.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))
            trace_entries.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "args": tool_call["args"],
                    "result": content,
                }
            )

        return {
            "messages": state["messages"] + tool_messages,
            "trace": state["trace"] + trace_entries,
            **state_updates,
        }

    async def _process_literature_claims_node(self, state: AgentState):
        """Node to process literature answers and extract/verify claims."""
        logger.info("Processing literature claims...")

        # Emit processing event
        self.step_counter += 1
        await self._emit_callback(
            {
                "type": "claim_processing_start",
                "content": "Extracting and verifying claims from literature...",
            }
        )

        try:
            # Get documents from state
            documents = state.get("literature_documents", [])

            if not documents:
                logger.warning("No documents found for claim processing")
                return state

            # Find ALL literature answers in recent messages (not just the first one!)
            # IMPORTANT: Only include answers that have documents (real literature results)
            # This filters out EHR query results that might be in the same tool output
            literature_answers = []
            for msg in reversed(state["messages"]):
                if isinstance(msg, ToolMessage):
                    try:
                        content = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                        for result in content.get("results", []):
                            # Only process results with documents (literature) - skip EHR results
                            if result.get("status") == "success" and result.get("result") and result.get("documents", []):  # Must have documents!
                                literature_answers.append(
                                    {
                                        "question": result.get("question", "Unknown question"),
                                        "answer": result["result"],
                                        "documents": result.get("documents", []),  # Include answer-specific documents!
                                    }
                                )
                    except Exception:
                        # Not a bare `except:` -- this runs inside an async node,
                        # and BaseException would swallow asyncio.CancelledError,
                        # making a cancelled or timed-out request continue looping.
                        continue

            if not literature_answers:
                logger.warning("No literature answers found for claim processing (answers without documents were filtered out)")
                return state

            logger.info(f"Processing claims for {len(literature_answers)} literature answer(s) (filtered to only include answers with documents)")
            logger.info(f"Total documents available: {len(documents)}")

            # Process ALL literature answers
            all_claims_results = []
            all_refinement_results = []
            all_verified_literature_results = []
            summary_msg = "\n\n═══════════════════════════════════════════════════════════════════\n"
            summary_msg += "               LITERATURE CLAIM VERIFICATION RESULTS\n"
            summary_msg += "═══════════════════════════════════════════════════════════════════\n\n"

            for idx, lit_answer_obj in enumerate(literature_answers, 1):
                literature_answer = lit_answer_obj["answer"]
                question = lit_answer_obj["question"]
                # CRITICAL: Use documents specific to THIS answer, not all documents!
                documents_for_this_answer = lit_answer_obj.get("documents", [])

                logger.info(f"Processing answer {idx}/{len(literature_answers)}: {question[:50]}...")
                logger.info(f"  Using {len(documents_for_this_answer)} documents specific to this answer")

                summary_msg += f"\n{'=' * 70}\n"
                summary_msg += f"LITERATURE ANSWER #{idx}: {question}\n"
                summary_msg += f"{'=' * 70}\n\n"

                # Process claims (not including partial by default as per requirements)
                try:
                    claims_result = await process_literature_answer_with_claims(
                        answer=literature_answer,
                        documents=documents_for_this_answer,  # Use answer-specific documents!
                        include_partial=False,
                    )
                    claims_result["question"] = question  # Add question to result
                except Exception as e:
                    logger.error(f"Failed to process claims for answer {idx}: {str(e)}")
                    # Create error result instead of crashing
                    claims_result = {
                        "question": question,
                        "original_answer": literature_answer,
                        "total_claims": 0,
                        "verified_claims": [],
                        "excluded_claims": [],
                        "error": str(e),
                    }

                all_claims_results.append(claims_result)

                # Check if claim extraction failed
                if "error" in claims_result:
                    summary_msg += "❌ CLAIM EXTRACTION FAILED\n"
                    summary_msg += f"Error: {claims_result['error'][:200]}\n\n"
                    summary_msg += "📋 Using original answer (unverified):\n"
                    summary_msg += f"{'-' * 70}\n"
                    summary_msg += f"{literature_answer[:500]}{'...' if len(literature_answer) > 500 else ''}\n"
                    summary_msg += f"{'-' * 70}\n\n"

                    # Create a placeholder refinement result
                    refinement_result = {
                        "question": question,
                        "refined_summary": literature_answer,
                        "kept_claims": 0,
                        "removed_claims": 0,
                        "modified_claims": 0,
                        "error": "Claim extraction failed",
                    }
                    all_refinement_results.append(refinement_result)
                    all_verified_literature_results.append(
                        {
                            "question": question,
                            "answer": literature_answer,
                            "verified_claims_result": [],
                        }
                    )
                    continue  # Skip to next answer

                # Refine the original summary based on claim verdicts
                logger.info(f"Refining answer {idx} based on claim verification verdicts...")
                try:
                    refinement_result = await refine_summary_with_claim_verdicts(original_summary=literature_answer, claims_data=claims_result)
                except Exception as e:
                    logger.error(f"Failed to refine answer {idx}: {str(e)}")
                    refinement_result = {
                        "question": question,
                        "refined_summary": literature_answer,
                        "kept_claims": 0,
                        "removed_claims": 0,
                        "modified_claims": 0,
                        "error": str(e),
                    }

                all_refinement_results.append(refinement_result)

                # Create a summary message about the claims
                verified_count = len(claims_result.get("verified_claims", []))
                excluded_count = len(claims_result.get("excluded_claims", []))
                partial_count = len([c for c in claims_result.get("excluded_claims", []) if c.get("attribution") == "PARTIALLY_ATTRIBUTABLE"])

                all_verified_literature_results.append(
                    {
                        "question": question,
                        "answer": refinement_result.get("refined_summary", literature_answer),
                        "verified_claims_result": claims_result["verified_claims"],
                    }
                )

                summary_msg += "📊 Claim Verification Summary:\n"
                summary_msg += f"  • Total claims extracted: {claims_result.get('total_claims', 0)}\n"
                summary_msg += f"  • ✅ Verified (ATTRIBUTABLE): {verified_count}\n"
                summary_msg += f"  • ⚠️  Partially attributable: {partial_count}\n"
                summary_msg += f"  • ❌ Excluded (UNRELATED/CONTRADICTORY): {excluded_count - partial_count}\n\n"

                # Add refined summary prominently
                if "error" not in refinement_result:
                    summary_msg += "✨ CORRECTED ANSWER (based on verified claims):\n"
                    summary_msg += f"{'-' * 70}\n"
                    summary_msg += f"{refinement_result['refined_summary']}\n"
                    summary_msg += f"{'-' * 70}\n\n"
                    summary_msg += "📈 Refinement Statistics:\n"
                    summary_msg += f"  • Claims kept: {refinement_result['kept_claims']}\n"
                    summary_msg += f"  • Claims removed: {refinement_result['removed_claims']}\n"
                    summary_msg += f"  • Partial claims (modified/dropped): {refinement_result['modified_claims']}\n\n"

                    if refinement_result.get("changes_made"):
                        summary_msg += f"📝 Changes made: {refinement_result['changes_made']}\n\n"
                else:
                    summary_msg += "⚠️  Summary refinement failed, using original answer\n"
                    summary_msg += f"Error: {refinement_result.get('error', 'Unknown')}\n\n"

                # Format verified claims with citations (for reference)
                if claims_result.get("verified_claims"):
                    formatted_answer = format_claims_as_answer(claims_result["verified_claims"])
                    summary_msg += f"📚 Individual Verified Claims (for reference):\n{formatted_answer}\n\n"

                # Show partially attributable claims for review
                partial_claims = [c for c in claims_result.get("excluded_claims", []) if c.get("attribution") == "PARTIALLY_ATTRIBUTABLE"]
                if partial_claims:
                    summary_msg += "⚠️  PARTIALLY ATTRIBUTABLE CLAIMS (Need Review):\n"
                    for i, claim in enumerate(partial_claims, 1):
                        refs = claim.get("references", [])
                        ref_indices = [str(ref["doc_index"]) for ref in refs[:10]]
                        ref_str = f"[{', '.join(ref_indices)}]" if ref_indices else ""
                        summary_msg += f"  {i}. {claim['claim']} {ref_str}\n"
                        summary_msg += f"     ↳ Reason: {claim.get('explanation', '')[:150]}...\n\n"

                # Show original answer for comparison
                summary_msg += "📋 Original Answer (before verification):\n"
                summary_msg += f"{'-' * 70}\n"
                summary_msg += f"{literature_answer[:500]}{'...' if len(literature_answer) > 500 else ''}\n"
                summary_msg += f"{'-' * 70}\n\n"

            # Store all results in state
            new_state = {
                **state,
                "literature_claims": all_claims_results if len(all_claims_results) == 1 else {"all_answers": all_claims_results},
                "refined_summary": all_refinement_results if len(all_refinement_results) == 1 else {"all_answers": all_refinement_results},
                "verified_literature_results": all_verified_literature_results
                if len(all_verified_literature_results) == 1
                else {"all_answers": all_verified_literature_results},
            }

            # Add summary to trace
            trace_entry = {
                "role": "system",
                "content": summary_msg,
                "claims_data": all_claims_results,
                "refinement_data": all_refinement_results,
                "answers_processed": len(literature_answers),
            }

            # Emit completion event
            total_verified = sum(len(cr.get("verified_claims", [])) for cr in all_claims_results)
            total_excluded = sum(len(cr.get("excluded_claims", [])) for cr in all_claims_results)

            await self._emit_callback(
                {
                    "type": "claim_processing_complete",
                    "verified_claims": total_verified,
                    "excluded_claims": total_excluded,
                    "answers_processed": len(literature_answers),
                    "refined": all("error" not in r for r in all_refinement_results),
                    "claims_data": all_claims_results,
                }
            )

            logger.info(f"Claim processing complete for {len(literature_answers)} answers: {total_verified} verified, {total_excluded} excluded")

            return {
                **new_state,
                "messages": state["messages"] + [AIMessage(content=summary_msg)],
                "trace": state["trace"] + [trace_entry],
            }

        except Exception as e:
            logger.error(f"Error during claim processing: {e}", exc_info=True)

            # Emit error event
            await self._emit_callback({"type": "claim_processing_error", "error": str(e)})

            # Return state unchanged on error
            return state

    async def _process_final_summary_claims_node(self, state: AgentState):
        """Node to verify claims in the final summary against all sources (literature + EHR)."""
        logger.info("Processing final summary claims...")

        # Emit processing event
        self.step_counter += 1
        await self._emit_callback(
            {
                "type": "final_summary_verification_start",
                "content": "Verifying claims in final summary against all sources...",
            }
        )

        try:
            # Step 1: Extract final summary from agent messages
            final_summary = extract_summary_section(state["messages"])

            if not final_summary:
                logger.warning("No final summary found in messages - skipping verification")
                return state

            logger.info(f"Extracted final summary ({len(final_summary)} chars)")
            logger.debug(f"Summary text preview (first 500 chars): {final_summary[:500]}")

            # Step 2: Prepare unified document list (literature + EHR)
            literature_docs = state.get("literature_documents", []) or []
            ehr_answers = state.get("ehr_answers", []) or []

            if not literature_docs and not ehr_answers:
                logger.warning("No documents or EHR answers available for verification")
                return {
                    **state,
                    "final_summary_text": final_summary,
                    "final_summary_claims": {"error": "No source documents available"},
                }

            logger.info(f"Preparing unified document list: {len(literature_docs)} literature + {len(ehr_answers)} EHR")
            unified_docs, ref_mapping = prepare_unified_document_list(literature_docs, ehr_answers)

            # Step 3: Verify final summary claims
            logger.info("Verifying final summary claims against unified documents...")
            verification_result = await process_final_summary_with_verification(
                final_summary=final_summary,
                unified_documents=unified_docs,
                reference_mapping=ref_mapping,
                include_partial=False,  # Don't include partial claims in final summary
            )

            # Step 4: Format results for display
            display_msg = prepare_final_summary_for_display(verification_result)

            # Emit completion event
            stats = verification_result["statistics"]
            await self._emit_callback(
                {
                    "type": "final_summary_verification_complete",
                    "total_claims": stats["total_claims"],
                    "verified_claims": stats["verified_claims"],
                    "excluded_claims": stats["excluded_claims"],
                    "failed_verifications": stats["failed_verifications"],
                }
            )

            logger.info(f"Final summary verification complete: {stats['verified_claims']}/{stats['total_claims']} claims verified")

            # Update state with verification results
            trace_entry = {
                "role": "system",
                "content": display_msg,
                "final_summary_verification": verification_result["statistics"],
            }

            return {
                **state,
                "unified_documents": unified_docs,
                "reference_mapping": ref_mapping,
                "final_summary_text": final_summary,
                "final_summary_claims": verification_result["verification"],
                "final_summary_refined": verification_result["refinement"],
                "messages": state["messages"] + [AIMessage(content=display_msg)],
                "trace": state["trace"] + [trace_entry],
            }

        except Exception as e:
            logger.error(f"Error during final summary verification: {e}", exc_info=True)

            # Emit error event
            await self._emit_callback({"type": "final_summary_verification_error", "error": str(e)})

            # Return state unchanged on error
            return state

    async def _execute_specific_tool(self, state: AgentState, tool_name: str):
        """Execute a specific tool by name."""
        last_message = state["messages"][-1]
        tasks = []

        for tool_call in last_message.tool_calls:
            if tool_call["name"] == tool_name:
                tool_to_call = next((t for t in self.tools if t.name == tool_call["name"]), None)
                if tool_to_call:
                    config = RunnableConfig(run_name=tool_call["name"])

                    # Always add database parameter to EHR-related tool calls - REQUIRED
                    tool_args = tool_call["args"].copy()
                    if tool_call["name"] in [
                        "batch_query_medical_data",
                        "query_medical_data",
                        "get_patient_count_scaling",
                    ]:
                        # Override database parameter - ensure it's always explicitly set
                        tool_args["database"] = self.database
                        logger.info(f"Tool call {tool_call['name']} will use database: {self.database}")

                    # Emit tool start event
                    self.step_counter += 1
                    await self._emit_callback(
                        {
                            "type": "tool_start",
                            "tool_name": tool_call["name"],
                            "arguments": tool_args,
                        }
                    )

                    tasks.append(tool_to_call.ainvoke(tool_args, config=config))

        if not tasks:
            return state  # No matching tool calls found

        tool_outputs = await asyncio.gather(*tasks, return_exceptions=True)
        tool_messages, trace_entries = [], []

        tool_call_index = 0
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == tool_name:
                output = tool_outputs[tool_call_index]
                content = f"Error: {output}" if isinstance(output, Exception) else str(output)

                # Emit tool complete or error event
                if isinstance(output, Exception):
                    await self._emit_callback(
                        {
                            "type": "tool_error",
                            "tool_name": tool_call["name"],
                            "error": str(output),
                        }
                    )
                else:
                    # Try to parse result for structured summary
                    result_summary = None
                    try:
                        parsed = json.loads(content) if isinstance(content, str) else content
                        if isinstance(parsed, dict):
                            result_summary = {
                                "status": parsed.get("status"),
                                "result_count": len(parsed.get("results", [])),
                                "has_errors": any(r.get("status") == "error" for r in parsed.get("results", [])),
                            }
                    except Exception as e:
                        logger.debug("Failed to parse tool result for summary: %s", e)

                    await self._emit_callback(
                        {
                            "type": "tool_complete",
                            "tool_name": tool_call["name"],
                            "result": content[:200] + "..." if len(content) > 200 else content,
                            "result_summary": result_summary,
                        }
                    )

                tool_messages.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))
                trace_entries.append({"role": "tool", "name": tool_call["name"], "args": tool_call["args"], "result": content})
                tool_call_index += 1

        return {"messages": state["messages"] + tool_messages, "trace": state["trace"] + trace_entries}

    @staticmethod
    def _get_content_str(msg) -> str:
        """Safely extract string content from a message (handles list content from Gemini)."""
        content = msg.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Gemini multi-modal format: list of dicts with "text" keys or plain strings
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
            return " ".join(parts)
        return str(content)

    @staticmethod
    def _fix_markdown(text: str) -> str:
        """Fix common Gemini markdown formatting issues.

        Handles:
        - Citations inserted mid-word: N[G5]on-Hispanic → Non-Hispanic [G5]
        - Missing newline before markdown headers: ...[G5]### 3. → ...[G5]\n\n### 3.
        - Asterisk bullet markers on same line as preceding text
        """

        # 1. Fix citations inserted mid-word.
        #    Pattern: a letter, then one or more [X#] refs, then a lowercase letter
        #    e.g. "N[G5]on" → "Non [G5]" / "I[G9][G5][G1]t" → "It [G9][G5][G1]"
        def _rejoin_word(m: re.Match) -> str:
            before = m.group("before")
            refs = m.group("refs")
            after = m.group("after")
            return f"{before}{after} {refs}"

        text = re.sub(
            r"(?P<before>[A-Za-z])(?P<refs>(?:\[[A-Z]\d+\])+)(?P<after>[a-z])",
            _rejoin_word,
            text,
        )

        # 2. Ensure markdown headers start on their own line
        text = re.sub(r"([^\n])(#{1,6}\s)", r"\1\n\n\2", text)

        # 3. Ensure bullet-style lines (* text) start on their own line
        #    but don't break already-correct lines or italic/bold markers
        text = re.sub(r"([^\n\s])\s*(\n?\*[A-Z])", r"\1\n\n\2", text)

        return text

    def _should_continue(self, state: AgentState):
        """Conditional edge to decide the next step."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            # Check if multiple DIFFERENT tools are being called simultaneously
            unique_tool_names = set(tc["name"] for tc in last_message.tool_calls)
            if len(unique_tool_names) > 1:
                logger.info(f"Multiple tools called simultaneously: {unique_tool_names} — routing to execute_all_tools")
                return "execute_all_tools"

            # Single tool type (possibly called multiple times)
            tool_call = last_message.tool_calls[0]
            tool_name = tool_call["name"]
            if tool_name in ["batch_query_medical_data", "query_medical_data"]:
                return "batch_query_medical_data"
            elif tool_name == "get_patient_count_scaling":
                return "get_patient_count_scaling"
            elif tool_name == "google_search_grounding":
                return "google_search_grounding"
            else:
                # Fallback for unknown tools
                return "agent"
        if isinstance(last_message, AIMessage):
            content = self._get_content_str(last_message).lower()
            # When summary is detected, route to final verification if enabled
            if "summary" in content or len(state["messages"]) > 20:
                # Enforce scaling: if EHR data was queried but scaling was not called,
                # loop back to the agent so it can call get_patient_count_scaling first.
                if "get_patient_count_scaling" in self.tool_names:
                    tool_names_used = set()
                    for msg in state["messages"]:
                        if isinstance(msg, ToolMessage) and hasattr(msg, "name") and msg.name:
                            tool_names_used.add(msg.name)
                        # Also check tool_call results in the trace
                        if hasattr(msg, "tool_calls"):
                            for tc in msg.tool_calls:
                                tool_names_used.add(tc["name"])

                    ehr_was_queried = "batch_query_medical_data" in tool_names_used
                    scaling_was_called = "get_patient_count_scaling" in tool_names_used
                    # Also check if main pipeline answer contains patient counts (combine phase)
                    has_ehr_data = ehr_was_queried or any(
                        isinstance(msg, HumanMessage) and isinstance(msg.content, str) and "main pipeline answer" in msg.content.lower()
                        for msg in state["messages"]
                    )

                    # Bounded. Routing back only helps if the model then calls
                    # the tool; when it will not -- and on the literature-only
                    # path it frequently will not, because combine() re-adds
                    # get_patient_count_scaling to a toolset that was overridden
                    # down to web search -- an unbounded retry cannot terminate.
                    # The graph then runs to recursion_limit and the caller gets
                    # an exception where the comparison should be. Ask a bounded
                    # number of times, then let the agent answer without scaling:
                    # a comparison missing a national extrapolation is worth more
                    # than no comparison at all.
                    # Scaling extrapolates a database count to a national one.
                    # A database with no population and no coverage figure has
                    # nothing to extrapolate to, so the model is right to skip
                    # the tool and enforcing it can only spin. Both shipped
                    # synthetic datasets are in that position by construction:
                    # they describe a generated population and represent no
                    # country. Ask only where the answer would mean something.
                    if has_ehr_data and not scaling_was_called and self._scaling_is_meaningful():
                        attempts = sum(1 for msg in state["messages"] if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None))
                        if attempts <= self.max_scaling_enforcement_attempts:
                            logger.warning(
                                "Scaling enforcement: EHR data present but get_patient_count_scaling was not called "
                                "(attempt %d/%d). Routing back to agent.",
                                attempts,
                                self.max_scaling_enforcement_attempts,
                            )
                            return "agent"
                        logger.warning(
                            "Scaling enforcement: get_patient_count_scaling still not called after %d attempts; continuing without it.",
                            self.max_scaling_enforcement_attempts,
                        )

                # If claim attribution is enabled, verify the final summary
                if self.enable_claim_attribution:
                    return "final_summary"
                else:
                    return "end"
        return "agent"

    def _scaling_is_meaningful(self) -> bool:
        """Whether this database has a population to extrapolate a count to."""
        from ascent_domain.omop.models.agents.langgraph.scaling_tools import get_database_metadata

        meta = get_database_metadata(self.database) or {}
        return bool(meta.get("population")) and bool(meta.get("estimated_coverage"))

    async def arun(
        self,
        question: str,
        custom_prompt: str | None = None,
        database: str | None = None,
        continue_from_state: AgentState | None = None,
    ) -> Dict[str, Any]:
        """Runs the agent on a given question.

        Args:
            question: The question to answer
            custom_prompt: Optional custom prompt template
            database: Database to query (required for initial runs, optional for continuations)
            continue_from_state: Optional previous agent state to continue from.
                               If provided, the agent will resume from this state
                               with a new prompt instead of starting fresh.

        Returns:
            Dictionary containing final_answer, execution_trace, and agent_state
        """
        # Update database if provided at runtime, but ensure it's never empty
        if database:
            if not database.strip():
                raise ValueError("Database parameter cannot be empty")
            self.database = database
            logger.info(f"Database updated to: {self.database}")

        # For initial runs, database is required
        if not continue_from_state and not self.database:
            raise ValueError("Database must be specified either during initialization or at runtime for initial runs")

        prompt_template = custom_prompt or literature_comparison_prompt
        user_message = f"Question: {question}"

        # Always include database information in the user message if we have one
        if self.database:
            user_message += f"\n\nDatabase to use for EHR queries: {self.database}"
            logger.info(f"Agent will execute with database: {self.database}")

        # Determine starting state
        if continue_from_state:
            # Continue from previous state with new prompt injected as SystemMessage
            logger.info("Continuing agent execution from previous state")
            new_messages = continue_from_state.get("messages", []) + [
                SystemMessage(content=prompt_template),
                HumanMessage(content=user_message),
            ]
            new_trace = continue_from_state.get("trace", []) + [
                {"role": "system", "content": prompt_template},
                {"role": "user", "content": user_message},
            ]
            initial_state: AgentState = {
                **continue_from_state,
                "messages": new_messages,
                "trace": new_trace,
            }
        else:
            # Start fresh — prompt as SystemMessage, question as HumanMessage
            logger.info("Starting new agent execution from scratch")
            initial_state: AgentState = {
                "messages": [
                    SystemMessage(content=prompt_template),
                    HumanMessage(content=user_message),
                ],
                "trace": [
                    {"role": "system", "content": prompt_template},
                    {"role": "user", "content": user_message},
                ],
                "literature_documents": None,
                "literature_claims": None,
                "refined_summary": None,
                "verified_literature_results": None,
                "ehr_answers": None,
                "unified_documents": None,
                "reference_mapping": None,
                "final_summary_text": None,
                "final_summary_claims": None,
                "final_summary_refined": None,
                "google_search_results": None,
            }

        try:
            config = RunnableConfig(recursion_limit=25)
            result = await self.app.ainvoke(initial_state, config=config)
            raw_answer = self._get_content_str(result["messages"][-1]) if result["messages"] else "No response generated"
            final_answer = self._fix_markdown(raw_answer)

            return_data = {
                "final_answer": final_answer,
                "execution_trace": result["trace"],
                "agent_state": {
                    "literature_documents": result.get("literature_documents"),
                    "literature_claims": result.get("literature_claims"),
                    "refined_summary": result.get("refined_summary"),
                    "verified_literature_results": result.get("verified_literature_results"),
                    "google_search_results": result.get("google_search_results"),
                    "ehr_answers": result.get("ehr_answers"),
                    "unified_documents": result.get("unified_documents"),
                    "reference_mapping": result.get("reference_mapping"),
                    "final_summary_text": result.get("final_summary_text"),
                    "final_summary_claims": result.get("final_summary_claims"),
                    "final_summary_refined": result.get("final_summary_refined"),
                },
                "agent_state_raw": result,  # Raw LangGraph state for continue_from_state chaining
            }

            # Include claims data if available
            if self.enable_claim_attribution and result.get("literature_claims"):
                return_data["literature_claims"] = result["literature_claims"]

                # Handle single or multiple answers
                claims_data = result["literature_claims"]
                if isinstance(claims_data, dict) and "all_answers" in claims_data:
                    # Multiple answers
                    all_claims = claims_data["all_answers"]
                    return_data["streamlit_data"] = {"all_answers": [prepare_claims_for_streamlit(c) for c in all_claims]}
                else:
                    # Single answer
                    return_data["streamlit_data"] = prepare_claims_for_streamlit(claims_data)

                # CRITICAL: Include refined summaries if available (check result state directly)
                # The refined_summary should be in the state from the claim processing node
                if result.get("refined_summary"):
                    logger.info("✅ Found refined_summary in result state")
                    refined_data = result["refined_summary"]
                    return_data["refined_summary"] = refined_data
                else:
                    logger.warning("⚠️  refined_summary not found in result state - check if claim processing node ran")
                    logger.debug(f"Available keys in result: {list(result.keys())}")

                # If we have refined_summary, use it to create the final answer
                if "refined_summary" in return_data:
                    refined_data = return_data["refined_summary"]

                    # Handle single or multiple refined summaries
                    if isinstance(refined_data, dict) and "all_answers" in refined_data:
                        # Multiple answers - concatenate all refined summaries
                        all_refined = refined_data["all_answers"]

                        # Separate successful and failed
                        successful_refinements = []
                        failed_refinements = []
                        for i, r in enumerate(all_refined):
                            if "error" not in r or r.get("kept_claims", 0) > 0:
                                successful_refinements.append((i, r))
                            else:
                                failed_refinements.append((i, r))

                        if successful_refinements or failed_refinements:
                            return_data["original_answer"] = final_answer

                            # Create a comprehensive answer with clear labels
                            refined_parts = []

                            for i, r in successful_refinements:
                                refined_parts.append(
                                    f"═══ LITERATURE ANSWER #{i + 1} (FACT-CHECKED) ═══\n"
                                    f"{r['refined_summary']}\n"
                                    f"[✅ {r.get('kept_claims', 0)} claims verified, "
                                    f"❌ {r.get('removed_claims', 0)} removed]\n"
                                )

                            for i, r in failed_refinements:
                                refined_parts.append(
                                    f"═══ LITERATURE ANSWER #{i + 1} (UNVERIFIED - Technical Error) ═══\n"
                                    f"{r.get('refined_summary', 'N/A')}\n"
                                    f"[⚠️ Claim extraction failed, showing original answer]\n"
                                )

                            return_data["final_answer"] = "\n\n".join(refined_parts)
                    else:
                        # Single answer
                        if "error" not in refined_data:
                            return_data["original_answer"] = final_answer
                            return_data["final_answer"] = (
                                f"═══ FACT-CHECKED ANSWER ═══\n"
                                f"{refined_data['refined_summary']}\n\n"
                                f"[✅ {refined_data.get('kept_claims', 0)} claims verified, "
                                f"❌ {refined_data.get('removed_claims', 0)} removed]"
                            )

                # Include final summary verification results if available
                if result.get("final_summary_refined"):
                    logger.info("✅ Found final_summary_refined in result state")
                    return_data["final_summary_refined"] = result["final_summary_refined"]
                    return_data["final_summary_claims"] = result.get("final_summary_claims")
                    return_data["unified_documents"] = result.get("unified_documents")
                    return_data["reference_mapping"] = result.get("reference_mapping")

                    # Use the refined final summary as the main answer
                    refined_final = result["final_summary_refined"]
                    if isinstance(refined_final, dict) and "refined_summary" in refined_final:
                        return_data["original_answer"] = final_answer
                        return_data["final_answer"] = refined_final["refined_summary"]
                        logger.info("✅ Using verified final summary as main answer")

            return return_data

        except Exception as e:
            # Report the failure as a failure. Putting the exception text in
            # final_answer made it indistinguishable from a real answer to every
            # caller, which is how a recursion-limit error reached a report as
            # the comparison narrative.
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            return {
                "final_answer": None,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_trace": initial_state["trace"],
            }

    async def combine(
        self,
        question: str,
        main_pipeline_answer: str,
        database: str,
        phase1_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Combine Phase 1 literature results with the main pipeline EHR answer.

        This method handles Phase 2 of the WS4 pipeline. It injects the main pipeline
        answer and database metadata into the agent state, makes get_patient_count_scaling
        available as a tool, and uses literature_combine_prompt with full instructions for
        scaling, prevalence calculations, and narrative style.

        Args:
            question: The original clinical question.
            main_pipeline_answer: The text answer from the main EHR pipeline.
            database: The OMOP database name.
            phase1_state: The raw agent state from Phase 1 (from arun's agent_state_raw).

        Returns:
            Dictionary containing final_answer, execution_trace, and agent_state.
        """
        # Look up database metadata for population context
        db_metadata = get_database_metadata(database)
        db_context = population_context_note(db_metadata)
        # Only a database that samples a real population can be scaled up to one.
        scalable = bool(db_metadata and db_metadata["estimated_coverage"] is not None and db_metadata["population"] is not None)

        # Build the combine prompt with main pipeline answer and metadata injected
        combine_user_message = (
            f'The user asked: "{question}"\n\n'
            f"## Main Pipeline Answer (ASCENT OMOP query on {database}) [A1]:\n"
            f"{main_pipeline_answer}\n"
            f"{db_context}\n\n"
            f"Using the literature and web search results from the previous phase, "
            f"and the EHR data above, write the combined summary following your instructions. "
            + (
                "Use get_patient_count_scaling to extrapolate the raw patient counts from the main "
                "pipeline answer to national estimates before writing your narrative."
                if scalable
                else "Do NOT extrapolate the raw patient counts to national estimates: this database "
                "covers no real population. Report the counts as they are and say plainly that the "
                "comparison with literature is illustrative only."
            )
        )

        # Ensure get_patient_count_scaling is available for the combine phase
        scaling_tool_available = any(t.name == "get_patient_count_scaling" for t in self.tools)
        if not scaling_tool_available:
            original_tools = self.tools
            self.tools = list(self.tools) + [get_patient_count_scaling]
            self.tool_names = {t.name for t in self.tools}
            self.llm_with_tools = self.llm.bind_tools(self.tools)
            # Rebuild graph with new tools
            self.app = self._build_graph()

        try:
            result = await self.arun(
                question=combine_user_message,
                custom_prompt=literature_combine_prompt,
                database=database,
                continue_from_state=phase1_state,
            )
            return result
        finally:
            # Restore original tools if we modified them
            if not scaling_tool_available:
                self.tools = original_tools
                self.tool_names = {t.name for t in self.tools}
                self.llm_with_tools = self.llm.bind_tools(self.tools)
                self.app = self._build_graph()
