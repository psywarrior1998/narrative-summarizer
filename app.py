import gradio as gr
from transformers import pipeline
import torch
import os

# Models available
MODEL_OPTIONS = [
    "facebook/bart-large-cnn",
    "google/long-t5-local-base",
    "mistralai/Mistral-7B-v0.1"
]

# Prompt options
PROMPT_OPTIONS = [
    "Bread only",
    "Butter only",
    "Bread and Butter"
]

def chunk_text_tokenwise(text, max_tokens=512, tokenizer=None):
    """Token-based chunking using tokenizer.encode"""
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided for token-based chunking")
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def apply_custom_prompt(chunk, prompt_type):
    if prompt_type == "Bread only":
        prompt = f"Transform the provided fictional narrative into a maximally compressed yet losslessly decompressible format optimized for LLM reconstruction. {chunk}"
    elif prompt_type == "Butter only":
        prompt = f"Solid foundation, but let's refine the granularity. Your 4-subpoint structure creates artificial symmetry where organic complexity should flourish. {chunk}"
    elif prompt_type == "Bread and Butter":
        # Apply both prompts sequentially
        prompt = f"Transform the provided fictional narrative into a maximally compressed yet losslessly decompressible format optimized for LLM reconstruction. {chunk}"
        prompt = f"Solid foundation, but let's refine the granularity. Your 4-subpoint structure creates artificial symmetry where organic complexity should flourish. {prompt}"
    else:
        prompt = chunk
    return prompt

def summarize_chunk(chunk, summarizer, prompt_type):
    prompt = apply_custom_prompt(chunk, prompt_type)
    summary = summarizer(prompt, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

def run_app(file_obj, model_name, prompt_type, iterations):
    if file_obj is None:
        return "❌ Please upload a valid text file."

    # Read file content safely
    try:
        text = file_obj.read().decode("utf-8")
    except Exception:
        return "❌ Unable to read the file. Please upload a valid UTF-8 encoded text file."

    # Load summarization pipeline & tokenizer
    summarizer = pipeline("summarization", model=model_name, device=0 if torch.cuda.is_available() else -1)
    tokenizer = summarizer.tokenizer

    # Chunk text by tokens
    chunks = chunk_text_tokenwise(text, max_tokens=512, tokenizer=tokenizer)

    # First pass summaries
    summaries = []
    for chunk in chunks:
        temp = chunk
        for _ in range(iterations):
            temp = summarize_chunk(temp, summarizer, prompt_type)
        summaries.append(temp)

    # Second pass: global summarization for compression
    combined_summary = " ".join(summaries)
    global_summary = combined_summary
    # Optionally do one more iteration on the combined summary for extra compression
    if iterations > 1:
        for _ in range(iterations - 1):
            global_summary = summarize_chunk(global_summary, summarizer, prompt_type)

    return global_summary

# Tips to guide the user on model choice
model_tip = gr.Markdown(
    """
    **Model Selection Tips:**
    - **facebook/bart-large-cnn:** Fast, general-purpose summarization for short to medium texts.
    - **google/long-t5-local-base:** Designed for long documents; better context handling.
    - **mistralai/Mistral-7B-v0.1:** High-quality nuanced summaries; resource-intensive.
    """
)

with gr.Blocks() as demo:
    gr.Markdown("# Narrative Summarizer")
    gr.Markdown("Upload your text file and select the summarization model and prompt type. Set the number of iterations for compression.")

    file_input = gr.File(label="Upload Text File (.txt)", file_types=['.txt'])
    model_dropdown = gr.Dropdown(choices=MODEL_OPTIONS, label="Choose Model", value=MODEL_OPTIONS[0])
    model_tip.render()
    prompt_dropdown = gr.Dropdown(choices=PROMPT_OPTIONS, label="Choose Prompt Type", value=PROMPT_OPTIONS[0])
    iterations_slider = gr.Slider(minimum=1, maximum=5, step=1, label="Iterations", value=1)
    output_text = gr.Textbox(label="Summary Output", lines=15)

    summarize_button = gr.Button("Summarize")
    summarize_button.click(
        fn=run_app,
        inputs=[file_input, model_dropdown, prompt_dropdown, iterations_slider],
        outputs=output_text
    )

demo.launch()
