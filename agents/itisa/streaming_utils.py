# itisa/streaming_utils.py
import gradio as gr
import re
from typing import Optional, Iterable

# Import necessary smolagents components
# Ensure these are compatible with your installed smolagents version
from smolagents.agent_types import AgentAudio, AgentImage, AgentText
from smolagents.agents import MultiStepAgent # For type hinting
from smolagents.memory import ActionStep, FinalAnswerStep, MemoryStep, PlanningStep


def get_step_footnote_content(step_log: MemoryStep, step_name: str) -> str:
    """
    Generates a footnote string for a memory step, including token counts and duration if available.

    Args:
        step_log (MemoryStep): The memory step object from the agent.
        step_name (str): A descriptive name for the step (e.g., "Tool Action", "Planning").

    Returns:
        str: An HTML formatted string for the footnote.
    """
    step_footnote = f"**{step_name}**"
    if hasattr(step_log, "input_token_count") and step_log.input_token_count is not None and \
       hasattr(step_log, "output_token_count") and step_log.output_token_count is not None:
        token_str = f" | Input tokens:{step_log.input_token_count:,} | Output tokens: {step_log.output_token_count:,}"
        step_footnote += token_str
    if hasattr(step_log, "duration") and step_log.duration is not None:
        step_duration = f" | Duration: {round(float(step_log.duration), 2)}s"
        step_footnote += step_duration
    # Return as simple markdown, Gradio will handle rendering
    return f"_{step_footnote}_"

def pull_messages_from_step(step_log: MemoryStep) -> Iterable[gr.ChatMessage]:
    """
    Extracts and formats agent execution steps into Gradio ChatMessage objects for streaming.

    Args:
        step_log (MemoryStep): The memory step object from the agent.

    Yields:
        gr.ChatMessage: Formatted chat messages representing agent thoughts, actions, and results.
    """
    if isinstance(step_log, ActionStep):
        step_number_str = f"Step {step_log.step_number}" if step_log.step_number is not None else "Tool Action"

        if hasattr(step_log, "model_output") and step_log.model_output is not None:
            model_output = str(step_log.model_output).strip()
            # Clean up common LLM output artifacts if necessary
            model_output = re.sub(r"```\s*<end_code>", "```", model_output)
            model_output = re.sub(r"<end_code>\s*```", "```", model_output)
            model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)
            model_output = model_output.strip()
            if model_output:
                yield gr.ChatMessage(role="assistant", content=f"Thinking...\n{model_output}", metadata={"message_type": "thought"})

        if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None:
            for tool_call in step_log.tool_calls:
                tool_name = tool_call.name
                tool_args = tool_call.arguments
                
                tool_args_display = ""
                if tool_name == "python_interpreter" and isinstance(tool_args, str):
                    # Display Python code nicely
                    tool_args_clean = re.sub(r"^```python\n", "", tool_args, 1)
                    tool_args_clean = re.sub(r"\n```$", "", tool_args_clean, 1)
                    tool_args_display = f"```python\n{tool_args_clean}\n```"
                elif isinstance(tool_args, dict):
                    args_str = "\n".join([f"  - {k}: {v}" for k, v in tool_args.items()])
                    tool_args_display = f"Arguments:\n{args_str}"
                else:
                    tool_args_display = str(tool_args)

                yield gr.ChatMessage(
                    role="assistant",
                    content=f"🛠️ Calling Tool: **{tool_name}**\n{tool_args_display}",
                    metadata={"message_type": "tool_call", "tool_name": tool_name}
                )

        if hasattr(step_log, "observations") and step_log.observations is not None and str(step_log.observations).strip():
            log_content = str(step_log.observations).strip()
            # Check for plot paths
            path_match = re.search(r"([results/plots/].*\.html)", log_content, re.IGNORECASE)
            if path_match:
                html_path_relative = path_match.group(1)
                yield gr.ChatMessage(role="assistant", content=f"📊 Plot generated: You can view it at `{html_path_relative}` (open this file locally).\nFull message: {log_content}", metadata={"message_type": "tool_result_plot"})
            else:
                yield gr.ChatMessage(role="assistant", content=f"📝 Tool Result:\n{log_content}", metadata={"message_type": "tool_result"})

        if hasattr(step_log, "error") and step_log.error is not None:
            yield gr.ChatMessage(role="assistant", content=f"💥 Error during action:\n{str(step_log.error)}", metadata={"message_type": "error"})
        
        # Optional: footnote for the step (can be verbose)
        # yield gr.ChatMessage(role="assistant", content=get_step_footnote_content(step_log, step_number_str), metadata={"message_type": "footnote"})


    elif isinstance(step_log, PlanningStep):
        if step_log.plan and str(step_log.plan).strip():
            yield gr.ChatMessage(role="assistant", content=f"📝 Planning...\n{str(step_log.plan)}", metadata={"message_type": "plan"})
        # Optional: footnote for planning (can be verbose)
        # yield gr.ChatMessage(role="assistant", content=get_step_footnote_content(step_log, "Planning"), metadata={"message_type": "footnote"})

    elif isinstance(step_log, FinalAnswerStep):
        final_answer = step_log.final_answer
        content_to_display = ""

        if isinstance(final_answer, AgentText):
            content_to_display = final_answer.to_string()
        elif isinstance(final_answer, (AgentImage, AgentAudio)):
            file_path = final_answer.to_string()
            mime_type = "image/png" # Default
            if isinstance(final_answer, AgentImage):
                if file_path.lower().endswith(".jpg") or file_path.lower().endswith(".jpeg"): mime_type = "image/jpeg"
                elif file_path.lower().endswith(".png"): mime_type = "image/png"
                # Add more image types if needed
            elif isinstance(final_answer, AgentAudio):
                if file_path.lower().endswith(".wav"): mime_type = "audio/wav"
                elif file_path.lower().endswith(".mp3"): mime_type = "audio/mpeg"
                # Add more audio types if needed
            yield gr.ChatMessage(role="assistant", content={"path": file_path, "mime_type": mime_type}, metadata={"message_type": "final_answer_file"})
            return # Exit after yielding file content

        else: # Fallback for other types or direct string answers
            content_to_display = str(final_answer)
        
        if content_to_display.strip():
            yield gr.ChatMessage(role="assistant", content=f"✅ Final Answer:\n{content_to_display}", metadata={"message_type": "final_answer"})

def stream_agent_responses(
    agent: MultiStepAgent,
    task: str,
    reset_agent_memory: bool = False,
    additional_args: Optional[dict] = None,
) -> Iterable[gr.ChatMessage]:
    """
    Runs an agent with the given task and streams the responses as Gradio ChatMessage objects.

    Args:
        agent (MultiStepAgent): The smolagent instance to run.
        task (str): The task or prompt for the agent.
        reset_agent_memory (bool): Whether to reset the agent's memory before running.
        additional_args (Optional[dict]): Additional arguments for the agent's run method.

    Yields:
        gr.ChatMessage: Formatted chat messages from the agent's execution steps.
    """
    try:
        for step_log in agent.run(task, stream=True, reset=reset_agent_memory, additional_args=additional_args):
            # Here, you could also update total token counts or other metadata if needed
            # For example, if step_log has token info:
            # if hasattr(step_log, "input_token_count"):
            #     # Update some global or session state counter
            #     pass

            for message_part in pull_messages_from_step(step_log):
                yield message_part
    except Exception as e:
        yield gr.ChatMessage(role="assistant", content=f"🚨 An error occurred while running the agent: {str(e)}", metadata={"message_type": "agent_error"})

