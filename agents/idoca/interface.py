"""Gradio user interface for IDOCA."""

import os
import time
import logging
from typing import List, Dict, Any, Optional, Set
from PIL import Image

import gradio as gr

from idoca.config import (
    DEFAULT_EMBEDDING_MODEL, DEFAULT_VISION_MODEL, DEFAULT_LLM_MODEL, 
    BOT_AVATAR_URL
)
from idoca.data_processor import DataProcessor
from idoca.rag import RAGSystem
from idoca.agent import IndustrialAgent
from idoca.utils import format_chatbot_message

logger = logging.getLogger("idoca.interface")

def create_interface():
    """Create the Gradio interface for IDOCA."""
    data_processor = DataProcessor()
    
    with gr.Blocks(title="Industrial Documents Analysis Agent", theme=gr.themes.Soft()) as demo:
        # State variables
        rag_s = gr.State(None)  # RAG system instance
        agent_s = gr.State(None)  # Agent instance
        pfi_s = gr.State([])  # Processed File Info (for DataFrame display) - Current batch
        pd_s = gr.State([])   # ALL Processed Documents (for RAG system) - Accumulates
        npd_s = gr.State([])  # Newly Processed Documents (since last init) - Buffer
        session_processed_paths_s = gr.State(set())  # Tracks unique file paths processed

        # CSS styling for the interface
        gr.HTML("""<style>
            /* Combined CSS styles */
            .gradio-container { font-family: 'IBM Plex Sans', sans-serif; }
            .gr-chatbot { font-size: 0.95rem; } .gr-chatbot .message-wrap { padding: 0.5rem; }
            .gr-chatbot .message-wrap .message { display: flex; flex-direction: column; padding: 0.6rem 0.75rem; border-radius: 0.5rem; margin-bottom: 0.5rem; box-shadow: 0 1px 2px 0 rgba(0,0,0,0.07); max-width: 95%; word-wrap: break-word; overflow-wrap: break-word; }
            .gr-chatbot .message-wrap .message.user { background-color: #E0E7FF; color: #3730A3; align-self: flex-end; margin-left: auto; }
            .gr-chatbot .message-wrap .message.bot { background-color: #F3F4F6; color: #1F2937 !important; align-self: flex-start; margin-right: auto; }
            .gr-chatbot .message-wrap .message.bot div[data-testid="bot"], .gr-chatbot .message-wrap .message.bot div[data-testid="bot"] * { color: #1F2937 !important; white-space: pre-wrap !important; word-wrap: break-word !important; font-size: 0.95rem; }
            .gr-chatbot .message-wrap .message.bot details { margin-top: 8px; border: 1px solid #D1D5DB; border-radius: 6px; padding: 8px; background-color: #E5E7EB;}
            .gr-chatbot .message-wrap .message.bot summary { font-weight: 600; cursor: pointer; color: #374151; margin-bottom: 4px;}
            /* Ensure preformatted text within details wraps correctly */
            .gr-chatbot .message-wrap .message.bot details > div pre, .gr-chatbot .message-wrap .message.bot details pre code { white-space: pre-wrap !important; word-wrap: break-word !important; background-color: #F9FAFB; padding: 6px; border-radius: 4px; display: block; }
            .jumping-dots span {display:inline-block;animation:blink 1.2s infinite;margin-left:1px;font-size:1.2em;} .jumping-dots .dot-1{animation-delay:0s;} .jumping-dots .dot-2{animation-delay:0.25s;} .jumping-dots .dot-3{animation-delay:0.5s;} @keyframes blink{0%,100%{opacity:0.2;}50%{opacity:1;}}
        </style>""")
        
        gr.Markdown("# 🏭 Industrial Documents Analysis Agent")

        with gr.Tabs() as tabs:
            # Tab 1: Upload & Initialize Data
            with gr.Tab("📚 Upload & Initialize Data", id=0):
                with gr.Row():
                    # Left column for controls
                    with gr.Column(scale=1):
                        gr.Markdown("#### **Step 1:** Upload Files")
                        unified_file_input = gr.File(
                            label="Select Docs & Images", 
                            file_count="multiple", 
                            file_types=[".pdf", ".csv", ".txt", ".md", ".jpg", ".jpeg", ".png"]
                        )
                        
                        gr.Markdown("#### **Step 2:** Select Models")
                        embedding_model_dd = gr.Dropdown(
                            choices=["nomic-embed-text", "all-MiniLM-L6-v2"], 
                            value=DEFAULT_EMBEDDING_MODEL, 
                            label="Embedding"
                        )
                        vision_model_dd = gr.Dropdown(
                            choices=["llava", "granite3.2-vision:2b-fp16"], 
                            value=DEFAULT_VISION_MODEL, 
                            label="Vision"
                        )
                        llm_model_dd = gr.Dropdown(
                            choices=["llama3", "qwen3:8b", "mistral"], 
                            value=DEFAULT_LLM_MODEL, 
                            label="LLM (RAG & Agent)"
                        )
                        
                        gr.Markdown("---")  # Separator
                        gr.Markdown("#### **Step 3:** Process Uploaded Files")
                        process_btn = gr.Button("Process New Files", variant="secondary")
                        
                        gr.Markdown("#### **Step 4:** Initialize Systems")
                        init_btn = gr.Button("🚀 Initialize/Re-Initialize RAG & Agent", variant="primary")
                        
                        gr.Markdown("---")  # Separator
                        clear_btn = gr.Button("Clear All Data & Session State", variant="stop")
                        global_status_out = gr.Textbox(
                            label="System Status", 
                            lines=2, 
                            interactive=False, 
                            placeholder="Current system status will appear here..."
                        )
                    
                    # Right column for logs and previews
                    with gr.Column(scale=2):
                        gr.Markdown("#### File Processing Log (Last Batch)")
                        doc_df = gr.Dataframe(
                            headers=["File", "Type", "Status", "Details"], 
                            label=" ", 
                            interactive=False, 
                            row_count=(7, "fixed")
                        )
                        gr.Markdown("#### Image Preview (Last Processed in Batch)")
                        with gr.Row():
                            img_prev_out = gr.Image(label=" ", type="pil", interactive=False, height=200)
                            img_desc_out = gr.Textbox(label="Description", lines=8, interactive=False)
            
            # Tab 2: Simple RAG
            with gr.Tab("🔍 Simple RAG", id=1):
                simple_chat = gr.Chatbot(
                    label="Simple RAG Conversation", 
                    height=550, 
                    type="messages",
                    show_copy_button=True, 
                    avatar_images=(None, BOT_AVATAR_URL)
                )
                with gr.Row():
                    simple_q_tb = gr.Textbox(
                        label="Your Question:", 
                        placeholder="Ask a question based on the processed documents...", 
                        lines=2, 
                        scale=4, 
                        container=False
                    )
                    simple_ask_btn = gr.Button("Ask", variant="primary", scale=1, min_width=100)
                gr.Examples(
                    examples=[
                        ["What is the maximum operational temperature for the electric arc furnace?"],
                        ["List all personal protective equipment required in the furnace area."],
                        ["What are the standard and premium grade chemical composition requirements for High Carbon Ferromanganese?"],
                        ["Summarize the complete emergency procedures for chemical spills, including contact numbers."]
                    ],
                    inputs=[simple_q_tb],
                    label="Example Questions (Simple RAG)"
                )
            
            # Tab 3: Agentic RAG
            with gr.Tab("🤖 Agentic RAG", id=2):
                agent_chat = gr.Chatbot(
                    label="Agentic RAG Conversation", 
                    height=550, 
                    type="messages",
                    show_copy_button=True, 
                    avatar_images=(None, BOT_AVATAR_URL)
                )
                with gr.Row():
                    agent_q_tb = gr.Textbox(
                        label="Your Question:", 
                        placeholder="Ask a complex question requiring reasoning or tools...", 
                        lines=2, 
                        scale=4, 
                        container=False
                    )
                    agent_ask_btn = gr.Button("Ask Agent", variant="primary", scale=1, min_width=100)
                gr.Examples(
                    examples=[
                        ["Convert the maximum operational EAF temperature of 1600°C to Fahrenheit. Is this temperature typical for manganese alloy production?"],
                        ["If the cooling water temperature increases from 32°C to 38°C, is this within operational limits? What specific actions should be taken according to procedures, and what would be the potential impact on furnace operation?"],
                        ["Based on the maintenance logs, what specific issues have been reported with the 'Backup Pump', and what were the exact actions taken to address them, including dates and technicians involved?"],
                        ["Compare the specifications for the Gas Cleaning System shown in the 'gas_cleaning_system.png' diagram (if available) with the operational requirements mentioned in the safety manual. Are there any discrepancies in parameters like pressure, emissions, or flow rates?"]
                    ],
                    inputs=[agent_q_tb],
                    label="Example Questions (Agentic RAG)"
                )
                gr.Markdown("**Agent Tools:** Knowledge Base Query, General Search, Numerical Analysis")

        # Handler functions
        def handle_process_files(
            files_list, 
            vision_model: str, 
            current_session_processed_paths_set: Optional[Set[str]]
        ) -> tuple:
            """Process uploaded files and prepare them for RAG."""
            current_batch_file_info_for_df = []
            newly_docs_for_rag_buffer = []
            console_log_messages = []

            if current_session_processed_paths_set is None:
                current_session_processed_paths_set = set()
                
            if not files_list:
                return (
                    "No files selected for processing.", 
                    current_batch_file_info_for_df, 
                    None, 
                    None, 
                    newly_docs_for_rag_buffer, 
                    current_session_processed_paths_set
                )

            latest_img_preview_for_ui, latest_img_desc_for_ui = None, None

            for f_obj in files_list:
                if f_obj is None: 
                    continue
                    
                temp_file_path, original_file_name = f_obj.name, os.path.basename(f_obj.name)
                file_extension = os.path.splitext(original_file_name)[1].lower()
                status_symbol, details_text, file_type_display = "❓", "Unknown", "Unknown"

                # Skip already processed files
                if temp_file_path in current_session_processed_paths_set:
                    status_symbol, details_text = "⏭️", "Already processed in this session"
                    file_type_display = "N/A (Skipped)"
                    console_log_messages.append(f"File '{original_file_name}': Skipped (already processed).")
                    current_batch_file_info_for_df.append([original_file_name, file_type_display, status_symbol, details_text])
                    continue
                    
                try:
                    # Process document files
                    if file_extension in ['.pdf', '.csv', '.txt', '.md']:
                        file_type_display = "Doc"
                        processed_chunks = data_processor.process_document_file(temp_file_path)
                        newly_docs_for_rag_buffer.extend(processed_chunks)
                        details_text, status_symbol = f"{len(processed_chunks)} chunks", "✅"
                    
                    # Process image files
                    elif file_extension in ['.jpg', '.jpeg', '.png']:
                        file_type_display = "Img"
                        image_document = data_processor.process_image_file(temp_file_path, vision_model)
                        newly_docs_for_rag_buffer.append(image_document)
                        details_text, status_symbol = "Description OK", "✅"
                        try:
                            latest_img_preview_for_ui = Image.open(temp_file_path)
                            latest_img_desc_for_ui = image_document.page_content
                        except Exception as img_e:
                            console_log_messages.append(f"Img '{original_file_name}': Preview UI err - {img_e}")
                    
                    # Handle unsupported file types
                    else:
                        details_text, status_symbol = f"Unsupported type ({file_extension})", "⚠️"
                    
                    console_log_messages.append(f"{file_type_display} '{original_file_name}': {status_symbol} ({details_text}).")
                    current_batch_file_info_for_df.append([original_file_name, file_type_display, status_symbol, details_text])
                    
                    if status_symbol == "✅":
                        current_session_processed_paths_set.add(temp_file_path)
                        
                except Exception as e:
                    status_symbol, details_text = "❌", f"Error: {str(e)[:40]}..."
                    console_log_messages.append(f"File '{original_file_name}': {status_symbol} - {e}")
                    current_batch_file_info_for_df.append([
                        original_file_name, 
                        file_type_display if file_type_display != "Unknown" else file_extension, 
                        status_symbol, 
                        details_text
                    ])
            
            final_console_status = "\n".join(console_log_messages)
            logger.info(f"--- File Processing Batch ---\n{final_console_status}\n---------------------------")
            
            ui_status_message = (
                f"{len(newly_docs_for_rag_buffer)} new doc(s)/chunk(s) processed. Ready for init."
                if newly_docs_for_rag_buffer else "No new files processed (already done or errors)."
            )
            
            return (
                ui_status_message, 
                current_batch_file_info_for_df, 
                latest_img_preview_for_ui, 
                latest_img_desc_for_ui, 
                newly_docs_for_rag_buffer, 
                current_session_processed_paths_set
            )

        def handle_initialize_systems(
            emb_model: str, 
            llm_model: str, 
            newly_docs_buffer: List, 
            current_rag_system: Optional[RAGSystem], 
            current_agent_system: Optional[IndustrialAgent], 
            all_docs_for_rag_accumulator: List
        ) -> tuple:
            """Initialize or reinitialize the RAG and Agent systems."""
            status_lines = ["--- System Initialization/Update ---"]
            
            # Add new documents to accumulator
            if newly_docs_buffer:
                if all_docs_for_rag_accumulator is None: 
                    all_docs_for_rag_accumulator = []
                all_docs_for_rag_accumulator.extend(newly_docs_buffer)
                status_lines.append(
                    f"ℹ️ Added {len(newly_docs_buffer)} new docs to total. " +
                    f"Total for RAG: {len(all_docs_for_rag_accumulator)}."
                )
            
            # Check if there are documents available
            if not all_docs_for_rag_accumulator:
                status_lines.append("❌ No documents available for RAG. Please process files first.")
                return "\n".join(status_lines), current_rag_system, current_agent_system, all_docs_for_rag_accumulator, []

            rag_instance, agent_instance = None, None
            
            # Initialize RAG system
            try:
                logger.info(f"Building/Rebuilding RAG with {len(all_docs_for_rag_accumulator)} total documents.")
                rag_instance = RAGSystem(embedding_model_name=emb_model, llm_model_name=llm_model)
                rag_instance.add_documents(all_docs_for_rag_accumulator)
                build_ok = rag_instance.build_vector_store(force_rebuild=True)
                chain_ok = build_ok and rag_instance.initialize_rag_chain()
                status_lines.extend(rag_instance.get_status(concise=True))
                
                if not chain_ok:
                    status_lines.append("❌ RAG system setup failed.")
                    rag_instance = None
            except Exception as e:
                status_lines.append(f"❌ RAG Setup Error: {e}")
                logger.error(traceback.format_exc())
                rag_instance = None

            # Initialize agent if RAG is available
            if rag_instance:
                try:
                    logger.info("Initializing/Re-initializing IndustrialAgent.")
                    agent_instance = IndustrialAgent(rag_system=rag_instance, llm_model_name=llm_model)
                    status_lines.extend(agent_instance.get_status(concise=True))
                    
                    if not agent_instance.agent_executor:
                        status_lines.append("⚠️ Agent graph compilation failed.")
                except Exception as e:
                    status_lines.append(f"❌ Agent Init Error: {e}")
                    logger.error(traceback.format_exc())
                    agent_instance = None
            else:
                agent_instance = None
                status_lines.append("ℹ️ Agent initialization skipped (RAG system not ready).")

            return "\n".join(status_lines), rag_instance, agent_instance, all_docs_for_rag_accumulator, []

        def handle_clear_data(current_rag_system: Optional[RAGSystem]):
            """Clear all data and reset state."""
            if current_rag_system:
                current_rag_system.clear_documents()
            
            # Reset all relevant states
            return None, None, [], [], [], [], None, None, "Status: All data & session state cleared.", set()

        def simple_rag_chat_fn(query: str, history: List[Dict], rag_system: Optional[RAGSystem]):
            """Handle user queries in the Simple RAG interface."""
            if not query.strip():
                return history
            
            # Create a new copy of the history list
            new_history = history.copy()
            
            # Add the user's message to the history
            user_message = format_chatbot_message("user", query)
            new_history.append(user_message)
            yield new_history  # Display user message immediately
            
            # Add a "thinking" message from the assistant
            thinking_message = format_chatbot_message(
                "assistant",
                "Thinking <span class='jumping-dots'><span class='dot-1'>.</span>" + 
                "<span class='dot-2'>.</span><span class='dot-3'>.</span></span>"
            )
            new_history.append(thinking_message)
            yield new_history  # Show thinking message
            
            # Check if the RAG system is initialized
            if rag_system is None or not rag_system.rag_chain:
                error_message = format_chatbot_message("assistant", "⚠️ RAG system not initialized.")
                new_history[-1] = error_message  # Replace thinking message with error
                yield new_history
                return
            
            try:
                # Query the RAG system
                result = rag_system.query(query)
                
                if result and 'error' in result:
                    # Handle error in RAG query
                    response_content = f"❌ Error: {result['error']}"
                elif result and 'result' in result:
                    # Process successful response
                    ans = result.get("result", "No answer.").replace("<think>", "").replace("</think>", "").strip()
                    response_content = ans
                    
                    # Add source information if available
                    srcs = []
                    if result.get("source_documents"):
                        for d_idx, d in enumerate(result["source_documents"][:2]):
                            pv = d.page_content[:70] + '...'
                            sn = os.path.basename(d.metadata.get('source', '?'))
                            doc_type_md = d.metadata.get("type", "document")
                            img_file_md = d.metadata.get('image_file', sn)
                            dt = "Img" if doc_type_md == "image_description" else "Doc"
                            ifn = img_file_md if dt == "Img" else sn
                            srcs.append(f"- {dt} *{ifn}* (match {d_idx + 1}): _{pv}_")
                        
                        if srcs:
                            response_content += "\n\n**Sources:**\n" + "\n".join(srcs)
                    
                    if not response_content.strip():
                        response_content = "Received an empty answer from RAG."
                else:
                    response_content = "Error in RAG query."
                
                # Replace the thinking message with the actual response
                response_message = format_chatbot_message("assistant", response_content)
                new_history[-1] = response_message
                yield new_history
                
            except Exception as e:
                logger.error(f"ERROR simple_rag_chat_fn: {e}\n{traceback.format_exc()}")
                error_message = format_chatbot_message("assistant", f"❌ Unexpected Error: {e}")
                new_history[-1] = error_message  # Replace thinking message with error
                yield new_history

        def agent_chat_fn(query: str, history: List[Dict], agent: Optional[IndustrialAgent]):
            """Handle user queries in the Agentic RAG interface."""
            if not query.strip():
                return history
            
            # Create a new copy of the history list
            new_history = history.copy()
            
            # Add the user's message to the history
            user_message = format_chatbot_message("user", query)
            new_history.append(user_message)
            yield new_history  # Display user message immediately
            
            # Add a "thinking" message from the assistant
            thinking_message = format_chatbot_message(
                "assistant",
                "Agent is thinking <span class='jumping-dots'><span class='dot-1'>.</span>" + 
                "<span class='dot-2'>.</span><span class='dot-3'>.</span></span>"
            )
            new_history.append(thinking_message)
            yield new_history  # Show thinking message
            
            # Check if agent is initialized
            if agent is None or not agent.agent_executor:
                error_message = format_chatbot_message("assistant", "⚠️ Agent system not ready.")
                new_history[-1] = error_message  # Replace thinking message with error
                yield new_history
                return
            
            try:
                stream_produced_output = False
                accumulated_text_for_current_bubble = ""
                current_bot_turn_messages = [thinking_message]  # Track messages in the current turn
                
                # Stream responses from the agent
                for i, msg_dict in enumerate(agent.stream_response(query)):
                    stream_produced_output = True
                    role = msg_dict.get("role", "assistant")
                    content = msg_dict.get("content", "")
                    metadata = msg_dict.get("metadata")
                    
                    formatted_msg_part = format_chatbot_message(role, content, metadata)
                    
                    if i == 0:  # First message from agent stream
                        # Replace the thinking message with this content
                        new_history[-1] = formatted_msg_part
                        current_bot_turn_messages = [formatted_msg_part]
                        if content and not metadata:  # If it's text content
                            accumulated_text_for_current_bubble = content
                    elif metadata:  # New metadata message (tool call, tool result) - add a new message
                        new_history.append(formatted_msg_part)
                        current_bot_turn_messages.append(formatted_msg_part)
                        accumulated_text_for_current_bubble = ""  # Reset for any potential next text bubble
                    elif content:  # Streaming more text content
                        if current_bot_turn_messages and not current_bot_turn_messages[-1].get("metadata"):
                            # Append to the last text bubble in the current turn
                            accumulated_text_for_current_bubble += content
                            updated_message = format_chatbot_message("assistant", accumulated_text_for_current_bubble)
                            new_history[-1] = updated_message  # Update the most recent message
                            current_bot_turn_messages[-1] = updated_message
                        else:
                            # This content starts a new text bubble
                            new_message = format_chatbot_message("assistant", content)
                            new_history.append(new_message)
                            current_bot_turn_messages.append(new_message)
                            accumulated_text_for_current_bubble = content
                    
                    yield new_history  # Show updated state
                
                # After stream finishes, provide a concluding message if needed
                if not stream_produced_output:
                    new_history[-1] = format_chatbot_message(
                        "assistant", 
                        "Agent process completed (no specific output generated)."
                    )
                elif current_bot_turn_messages and current_bot_turn_messages[0].get("content", "").startswith("Agent is thinking"):
                    new_history[-1] = format_chatbot_message(
                        "assistant", 
                        "Agent process completed (stream ended early)."
                    )
                elif current_bot_turn_messages and current_bot_turn_messages[-1].get("metadata") and not current_bot_turn_messages[-1].get("content", "").strip():
                    # Last message was a tool log with no text after it
                    new_history.append(format_chatbot_message(
                        "assistant", 
                        "Agent has completed its actions. You can ask another question or refine your query."
                    ))
                elif current_bot_turn_messages and not current_bot_turn_messages[-1].get("content", "").strip() and not current_bot_turn_messages[-1].get("metadata"):
                    # Last message is an empty content bubble, provide a generic completion
                    new_history[-1]["content"] = "Agent process completed."  # Avoid empty bubble
                
                yield new_history
                
            except Exception as e:
                logger.error(f"ERROR agent_chat_fn: {e}\n{traceback.format_exc()}")
                error_message = format_chatbot_message("assistant", f"❌ Agent Error: {e}")
                
                # Replace the thinking message or append error if we're beyond that point
                if new_history[-1].get("content", "").startswith("Agent is thinking"):
                    new_history[-1] = error_message
                else:
                    new_history.append(error_message)
                
                yield new_history

        # Setup event listeners
        process_btn.click(
            fn=handle_process_files, 
            inputs=[unified_file_input, vision_model_dd, session_processed_paths_s],
            outputs=[global_status_out, doc_df, img_prev_out, img_desc_out, npd_s, session_processed_paths_s]
        )

        init_btn.click(
            fn=handle_initialize_systems,
            inputs=[embedding_model_dd, llm_model_dd, npd_s, rag_s, agent_s, pd_s],
            outputs=[global_status_out, rag_s, agent_s, pd_s, npd_s]
        )

        clear_btn.click(
            fn=handle_clear_data, 
            inputs=[rag_s],
            outputs=[
                rag_s, agent_s, pd_s, npd_s, pfi_s, doc_df, img_prev_out, img_desc_out,
                global_status_out, session_processed_paths_s
            ]
        ).then(lambda: ([], []), outputs=[simple_chat, agent_chat])

        simple_ask_btn.click(
            fn=simple_rag_chat_fn, 
            inputs=[simple_q_tb, simple_chat, rag_s], 
            outputs=[simple_chat]
        ).then(
            lambda: gr.update(value=""), 
            outputs=[simple_q_tb]
        )

        simple_q_tb.submit(
            fn=simple_rag_chat_fn, 
            inputs=[simple_q_tb, simple_chat, rag_s], 
            outputs=[simple_chat]
        ).then(
            lambda: gr.update(value=""), 
            outputs=[simple_q_tb]
        )

        agent_ask_btn.click(
            fn=agent_chat_fn, 
            inputs=[agent_q_tb, agent_chat, agent_s], 
            outputs=[agent_chat]
        ).then(
            lambda: gr.update(value=""), 
            outputs=[agent_q_tb]
        )

        agent_q_tb.submit(
            fn=agent_chat_fn, 
            inputs=[agent_q_tb, agent_chat, agent_s], 
            outputs=[agent_chat]
        ).then(
            lambda: gr.update(value=""), 
            outputs=[agent_q_tb]
        )

    return demo