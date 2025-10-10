from pathlib import Path
from langchain_core.runnables.graph import MermaidDrawMethod
from typing import Any


def save_workflow_graph_png(workflow: Any) -> None:
    """
    Save the workflow graph visualization as a PNG in the tools directory.
    Args:
        workflow: The compiled workflow object with get_graph().draw_mermaid_png().
    """
    filename = "workflow_graph.png"
    # Always save to the tools directory
    tools_dir = Path(__file__).parent
    png_bytes = workflow.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    out_path = tools_dir / filename
    out_path.write_bytes(png_bytes)
