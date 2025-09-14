import gradio as gr
from transformers import pipeline
import torch
import os
import chardet
from summarizer import NarrativeSummarizer # <-- Import the core logic

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

def run_app(file_obj, text_input, model_name, prompt_type, iterations, progress=gr.Progress()):
    # Determine the input source: file or direct text
    if file_obj is not None:
        try:
            with open(file_obj.name, 'rb') as f:
                raw_data = f.read()
            detected = chardet.detect(raw_data)
            encoding = detected['encoding'] or 'utf-8'
            text = raw_data.decode(encoding, errors='replace')
        except Exception as e:
            return f"❌ Unable to read the file: {str(e)}"
    elif text_input:
        text = text_input
    else:
        return "❌ Please upload a file or enter text to summarize."

    # Instantiate the summarizer and process the text
    try:
        summarizer = NarrativeSummarizer(model_name=model_name)
        
        # --- Start of new progress tracking logic ---
        # Chunk the text first to get the total number of chunks
        chunks = summarizer.chunk_text_tokenwise(text, max_tokens=512, overlap=50)
        total_chunks = len(chunks)
        condensed_chunks = []
        progress(0, desc="Loading Model...")

        # First pass summarization over all chunks
        progress(0, desc="Summarizing Chunks (Pass 1 of 1)...")
        for i, chunk in enumerate(chunks):
            progress((i + 1) / total_chunks, desc=f"Processing Chunk {i + 1}/{total_chunks}...")
            temp_chunk = chunk
            for _ in range(iterations):
                temp_chunk = summarizer.summarize_chunk(temp_chunk, prompt_type)
            condensed_chunks.append(temp_chunk)

        # Second pass for global compression
        combined = " ".join(condensed_chunks)
        # We perform a final summary only if the combined text is large, for efficiency
        if len(combined.split()) > summarizer.tokenizer.model_max_length * 0.8:
            progress(1.0, desc="Performing Final Summarization Pass...")
            final_summary = summarizer.summarize_chunk(combined, prompt_type)
        else:
            final_summary = combined

        progress(1.0, desc="Summarization Complete!")
        # --- End of new progress tracking logic ---

        return final_summary
    except Exception as e:
        return f"An error occurred during summarization: {str(e)}"

# Gradio Interface
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
    gr.Markdown("Upload your text file OR enter text below to get a summarized version.")

    with gr.Row():
        file_input = gr.File(label="Upload Text File (.txt)", file_types=['.txt'])
        text_input = gr.Textbox(label="Or, paste your text here", lines=10)

    with gr.Row():
        model_dropdown = gr.Dropdown(choices=MODEL_OPTIONS, label="Choose Model", value=MODEL_OPTIONS[0])
        prompt_dropdown = gr.Dropdown(choices=PROMPT_OPTIONS, label="Choose Prompt Type", value=PROMPT_OPTIONS[0])
        iterations_slider = gr.Slider(minimum=1, maximum=5, step=1, label="Iterations", value=1)

    model_tip.render()

    summarize_button = gr.Button("Summarize")
    output_text = gr.Textbox(label="Summary Output", lines=15)

    summarize_button.click(
        fn=run_app,
        inputs=[file_input, text_input, model_dropdown, prompt_dropdown, iterations_slider],
        outputs=output_text
    )

demo.launch()