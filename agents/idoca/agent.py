"""LLM-based agent implementation for IDOCA."""

import os
import re
import logging
import traceback
import time
from typing import List, Dict, Any, Optional, Iterator

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from idoca.config import DEFAULT_LLM_MODEL
from idoca.utils import MockChatModel
from idoca.rag import RAGSystem

logger = logging.getLogger("idoca.agent")

class AgentState(MessagesState):
    """State container for the agent workflow."""
    pass

class IndustrialAgent:
    """Implements an LLM-powered agent with specialized tools for industrial document analysis."""
    
    def __init__(self, rag_system: RAGSystem, llm_model_name: str = DEFAULT_LLM_MODEL, temperature: float = 0.0):
        self.rag_system = rag_system
        self.llm_model_name = llm_model_name
        self.temperature = temperature
        self.agent_llm = None
        self.agent_executor = None
        self.tools: List[Any] = []
        self.status_messages: List[str] = []
        self._initialize_agent_llm()
        self._setup_tools()
        self._compile_graph()

    def _initialize_agent_llm(self):
        """Initialize the agent's LLM with fallback to mock implementation."""
        try:
            self.agent_llm = ChatOllama(model=self.llm_model_name, temperature=self.temperature)
            self.status_messages.append(f"✅ AgentLLM:'{self.llm_model_name}' OK.")
            logger.info(f"AgentLLM '{self.llm_model_name}' OK.")
        except Exception as e:
            self.agent_llm = MockChatModel()
            self.status_messages.append(f"⚠️ AgentLLM:MOCK ('{self.llm_model_name}' fail:{e}).")
            logger.warning(f"AgentLLM MOCK.")

    def _setup_tools(self):
        """Create and register tools for the agent."""
        if not self.rag_system:
            logger.error("RAG missing for Agent. KB tool skipped.")
            
        logger.info("Setting up agent tools...")
        defined_tools = []
        
        # Tool 1: Knowledge Base Query (if RAG is available)
        if self.rag_system:
            @tool
            def query_document_knowledge_base(query: str) -> str:
                """Searches uploaded documents/images. Use for specific details, procedures, specs within provided materials."""
                if not self.rag_system.rag_chain:
                    return "⚠️ RAG chain N/A for KB query."
                try:
                    logger.info(f"Tool 'KB_query':'{query}'")
                    result = self.rag_system.query(query)
                    if result and 'error' in result:
                        return f"Err KB:{result['error']}"
                    elif result and 'result' in result:
                        ans = result.get("result", "No KB info.")
                        srcs = [os.path.basename(d.metadata.get('source', '?')) 
                               for d in result.get('source_documents', [])[:2]]
                        return f"{ans[:1000]} (Srcs:{', '.join(srcs) if srcs else 'N/A'})"
                    return "No info/bad format from RAG."
                except Exception as e:
                    logger.error(f"Tool KB query:{e}")
                    return f"Err KB:{e}"
            defined_tools.append(query_document_knowledge_base)

        # Tool 2: General Information Search
        @tool
        def search_for_general_information(query: str) -> str:
            """Simulates external web search for broad industrial info, definitions, context NOT in uploaded docs."""
            logger.info(f"Tool 'general_search':'{query}'")
            lc = query.lower()
            if "eaf" in lc and "temperature" in lc: 
                return "Sim:EAF Mn alloy temps 1400-1600°C."
            if "ppe" in lc and "furnace" in lc: 
                return "Sim:PPE furnace:hard hats,shields,heat-gear,boots,respirators."
            if "manganese alloy" in lc and "uses" in lc: 
                return "Sim:Mn alloys in steel(deoxidizers,alloying)&cast iron."
            return f"Sim search for '{query}':General info..."
        defined_tools.append(search_for_general_information)

        # Tool 3: Numerical Parameter Analysis
        @tool
        def analyze_numerical_parameters(parameter_query: str) -> str:
            """Analyzes numerical data. Converts C<>F, bar<>psi<>kPa, or gives context(e.g.,'Is 1500C high?')."""
            logger.info(f"Tool 'numerical_analysis':'{parameter_query}'")
            q = parameter_query.lower()
            
            # Handle temperature conversion
            cm = re.search(r'(-?\d+\.?\d*)\s*°?c', q)
            fm = re.search(r'(-?\d+\.?\d*)\s*°?f', q)
            if cm and ('to f' in q or 'fahrenheit' in q):
                c_val, f_val = float(cm.group(1)), (float(cm.group(1)) * 9 / 5) + 32
                return f"{c_val}°C is {f_val:.1f}°F."
            if fm and ('to c' in q or 'celsius' in q):
                f_val, c_val = float(fm.group(1)), (float(fm.group(1)) - 32) * 5 / 9
                return f"{f_val}°F is {c_val:.1f}°C."
            
            # Handle pressure conversion
            pm = re.search(r'(\d+\.?\d*)\s*(bar|psi|kpa|mbar)', q)
            if pm:
                v_val, u_val = float(pm.group(1)), pm.group(2)
                if u_val == "bar": 
                    return f"{v_val}bar≈{v_val * 100:.1f}kPa≈{v_val * 14.5:.1f}psi.(1bar≈atm)."
                if u_val == "kpa": 
                    return f"{v_val}kPa≈{v_val / 100:.2f}bar≈{v_val * 0.145:.2f}psi.(101.3kPa≈atm)."
                if u_val == "psi": 
                    return f"{v_val}psi≈{v_val / 14.5:.2f}bar≈{v_val * 6.89:.2f}kPa.(Tires~30-40psi)."
                if u_val == "mbar": 
                    return f"{v_val}mbar={v_val / 10:.1f}kPa.(Neg mbar=vacuum)."
            
            # Handle common contextual questions about numerical values
            if '1600' in q and '°c' in q: 
                return "1600°C very high, EAF smelting."
            if '35' in q and 'mw' in q: 
                return "35MW high power, EAFs."
            if '-10' in q and 'mbar' in q: 
                return "-10mbar slight vacuum, furnace extraction."
            if '10' in q and 'mg/nm' in q: 
                return "10mg/Nm³ emission limit particulates."
                
            return f"Num.analysis for '{parameter_query}':No rule."
            
        defined_tools.append(analyze_numerical_parameters)
        
        self.tools = defined_tools
        self.status_messages.append(f"✅ AgentTools:{len(self.tools)} def.")
        logger.info(f"{len(self.tools)} tools:{[t.name for t in self.tools]}")

    def _compile_graph(self):
        """Create and compile the agent workflow graph."""
        if not self.agent_llm:
            self.status_messages.append("❌ AgentGraph:LLM N/A.")
            logger.error("AgentGraph:LLM N/A.")
            return
            
        if not self.tools:
            self.status_messages.append("⚠️ AgentGraph:No tools.")
            logger.warning("AgentGraph:No tools.")
            
        logger.info("Compiling agent graph...")
        
        try:
            # Create LLM with tool binding if tools are available
            runnable = self.agent_llm.bind_tools(self.tools) if self.tools else self.agent_llm
            
            # Define agent node that processes messages
            def agent_node(state: AgentState):
                return {"messages": [runnable.invoke(state["messages"])]}
            
            # Create graph and add nodes
            graph = StateGraph(AgentState)
            graph.add_node("agent", agent_node)
            
            # Configure graph with or without tools
            if self.tools:
                graph.add_node("tools", ToolNode(self.tools))
                graph.set_entry_point("agent")
                graph.add_conditional_edges(
                    "agent", 
                    tools_condition, 
                    {"tools": "tools", END: END}
                )
                graph.add_edge("tools", "agent")
            else:
                graph.set_entry_point("agent")
                graph.add_edge("agent", END)
                
            # Compile graph
            self.agent_executor = graph.compile()
            self.status_messages.append(f"✅ AgentGraph:OK ({'w/' if self.tools else 'no'}tools).")
            logger.info("AgentGraph OK.")
            
        except Exception as e:
            self.status_messages.append(f"❌ AgentGraph:Fail-{e}")
            logger.error(f"AgentGraph Fail:{e}\n{traceback.format_exc()}")
            self.agent_executor = None

    def stream_response(self, query: str, thread_id: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """Stream the agent's response to a query, including tool calls and results."""
        if not self.agent_executor:
            yield {"role": "assistant", "content": "⚠️ Agent N/A."}
            return
            
        # Create thread ID for this interaction
        tid = thread_id or f"agent_thread_{time.time_ns()}"
        graph_input = {"messages": [HumanMessage(content=query)]}
        config = {"recursion_limit": 15, "configurable": {"thread_id": tid}}
        logger.info(f"Agent stream q:'{query}' (Th:{tid})")
        
        try:
            # Process graph execution events
            for chunk in self.agent_executor.stream(graph_input, config=config):
                # Handle agent outputs (reasoning, final response)
                if "agent" in chunk:
                    msgs = chunk["agent"].get("messages", [])
                    if msgs and isinstance(msgs[-1], AIMessage):
                        last_msg = msgs[-1]
                        if last_msg.tool_calls:
                            # Show tool planning
                            details = [
                                f"Tool:**{tc.get('name')}**\nArgs:`{str(tc.get('args', {}))[:100]}...`" 
                                for tc in last_msg.tool_calls
                            ]
                            yield {
                                "role": "assistant", 
                                "content": "", 
                                "metadata": {
                                    "title": "🧠 Agent:Planning Tool(s)", 
                                    "log": "\n---\n".join(details)
                                }
                            }
                        elif last_msg.content:
                            # Show agent's content response
                            yield {"role": "assistant", "content": str(last_msg.content)}
                
                # Handle tool execution results            
                elif "tools" in chunk:
                    for tool_msg in chunk["tools"].get("messages", []):
                        if isinstance(tool_msg, ToolMessage):
                            log = f"{str(tool_msg.content)[:300]}..."
                            yield {
                                "role": "assistant", 
                                "content": "", 
                                "metadata": {
                                    "title": f"🛠️ Agent:Output from '{tool_msg.name}'", 
                                    "log": log
                                }
                            }
        except Exception as e:
            logger.error(f"stream_response:{e}\n{traceback.format_exc()}")
            yield {"role": "assistant", "content": f"❌ Agent stream error:{e}"}

    def get_status(self, concise=False) -> List[str]:
        """Get the current status of the agent components."""
        llm_ok = isinstance(self.agent_llm, ChatOllama)
        graph_ok = bool(self.agent_executor)
        
        s = (f"Agent:{'OK' if llm_ok and graph_ok else 'Needs Attention'} "
             f"(LLM:{'OK' if llm_ok else 'F'},Tools:{len(self.tools)},Graph:{'OK' if graph_ok else 'F'})")
             
        if concise:
            return [s]
            
        ls = f"Ollama '{self.llm_model_name}'" if llm_ok else ("Mock" if isinstance(self.agent_llm, MockChatModel) else "NL")
        
        return [
            f"--- Agent Status ---", 
            f"LLM:{ls}", 
            f"Tools:{len(self.tools)}",
            f"Graph:{'OK' if graph_ok else 'Not OK'}"
        ] + self.status_messages[-1:]