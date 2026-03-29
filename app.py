
import os
from typing import Any, Dict, List

import gradio as gr


def _format_context(chunks: List[Dict[str, Any]], max_items: int = 3) -> str:
    if not chunks:
        return "No context chunks returned."

    lines = []
    for i, chunk in enumerate(chunks[:max_items], start=1):
        text = str(chunk.get("text", "")).strip()
        source = str(chunk.get("source", "unknown"))
        preview = (text[:350] + "...") if len(text) > 350 else text
        lines.append(f"[{i}] Source: {source}\n{preview}")
    return "\n\n".join(lines)


def run_rag(query: str):
    query = (query or "").strip()
    if not query:
        return "Please enter a question.", "", "", ""

    # Placeholder behavior until notebook pipeline is moved into importable Python modules.
    # Replace this block with actual pipeline calls, e.g. run_single_query_pipeline(query, retriever).
    example_output = {
        "answer": "Connect this Gradio app to your RAG pipeline functions in Python modules.",
        "retrieved_chunks": [],
        "metrics": {
            "faithfulness": "N/A",
            "relevancy": "N/A",
        },
    }

    answer = str(example_output.get("answer", ""))
    chunks = example_output.get("retrieved_chunks", [])
    metrics = example_output.get("metrics", {})

    context_preview = _format_context(chunks)
    faithfulness = str(metrics.get("faithfulness", "N/A"))
    relevancy = str(metrics.get("relevancy", "N/A"))
    return answer, context_preview, faithfulness, relevancy


with gr.Blocks(title="RAG Assignment 3") as demo:
    gr.Markdown("# RAG-based Question Answering System")
    gr.Markdown(
        "Set environment variables: HF_TOKEN, PINECONE_API_KEY, PINECONE_ENVIRONMENT"
    )

    query_input = gr.Textbox(label="Ask a question", lines=2, placeholder="Type your question here...")
    submit_btn = gr.Button("Generate Answer", variant="primary")

    answer_output = gr.Textbox(label="Generated Answer", lines=8)
    context_output = gr.Textbox(label="Retrieved Context (Top Chunks)", lines=10)
    faithfulness_output = gr.Textbox(label="Faithfulness Score")
    relevancy_output = gr.Textbox(label="Relevancy Score")

    submit_btn.click(
        fn=run_rag,
        inputs=[query_input],
        outputs=[answer_output, context_output, faithfulness_output, relevancy_output],
    )
    query_input.submit(
        fn=run_rag,
        inputs=[query_input],
        outputs=[answer_output, context_output, faithfulness_output, relevancy_output],
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
