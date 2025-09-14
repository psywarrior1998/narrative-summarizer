import gradio as gr
import torch
import os
import chardet
import time
from core.summarizer import NarrativeSummarizer # <-- Import the core logic

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
    "Bread and Butter",
    "Custom Prompt"
]

def run_app(file_obj, text_input, model_name, local_model_path, prompt_type, custom_prompt_text, iterations, batch_size, target_word_count, progress=gr.Progress()):
    start_time = time.time()
    
    # Check if custom prompt is selected but not provided
    if prompt_type == "Custom Prompt" and not custom_prompt_text:
        return "❌ Error: 'Custom Prompt' selected but no custom prompt provided.", "", "", None

    # Determine the input source: file or direct text
    if file_obj is not None:
        progress(0, desc="Reading file and detecting encoding...")
        try:
            with open(file_obj.name, 'rb') as f:
                raw_data = f.read()
            detected = chardet.detect(raw_data)
            encoding = detected['encoding'] or 'utf-8'
            text = raw_data.decode(encoding, errors='replace')
        except Exception as e:
            return f"❌ Unable to read the file: {str(e)}", "", "", None
    elif text_input:
        text = text_input
    else:
        return "❌ Please upload a file or enter text to summarize.", "", "", None

    input_word_count = len(text.split())
    
    # Override model name with local path if provided
    actual_model_name = local_model_path if local_model_path else model_name

    # Instantiate the summarizer and process the text
    try:
        progress(0.1, desc=f"Loading model: {actual_model_name}...")
        summarizer = NarrativeSummarizer(model_name=actual_model_name)
        
        chunks = summarizer.chunk_text_tokenwise(text, max_tokens=512, overlap=50)
        total_chunks = len(chunks)
        
        log_messages = [
            f"✅ Input ready. Word Count: {input_word_count}",
            f"✅ Using model: {actual_model_name}",
            f"✅ Split into {total_chunks} chunks.",
            f"✅ Beginning summarization with {iterations} passes..."
        ]

        condensed_chunks = []
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            progress(i / total_chunks, desc=f"Processing batch {i // batch_size + 1} of {total_chunks // batch_size + 1}...")
            
            # This is where the actual summarization happens for a single batch
            for _ in range(iterations):
                batch_summaries = summarizer.summarize_batch(
                    batch, 
                    prompt_type, 
                    custom_prompt_text if prompt_type == "Custom Prompt" else None,
                    target_word_count
                )
                batch = batch_summaries
            condensed_chunks.extend(batch)
        
        log_messages.append(f"✅ Pass 1 complete. Combining summaries...")

        # Second pass for global compression
        combined = " ".join(condensed_chunks)
        
        final_summary = combined
        # Check if the combined text is large enough for a final summary
        if len(summarizer.tokenizer.encode(combined)) > summarizer.tokenizer.model_max_length * 0.8:
            log_messages.append("✅ Final text is large, performing global summarization...")
            final_summary = summarizer.summarize_batch(
                [combined], 
                prompt_type, 
                custom_prompt_text if prompt_type == "Custom Prompt" else None,
                target_word_count
            )[0]

        end_time = time.time()
        duration = round(end_time - start_time, 2)
        
        summary_word_count = len(final_summary.split())
        log_messages.append(f"✅ Summarization complete in {duration} seconds.")
        log_messages.append(f"✅ Final Summary Word Count: {summary_word_count}")

        # Gradio now handles file downloads
        output_file_path = "summary.txt"
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(final_summary)

        return final_summary, "✅\n" + "\n".join(log_messages), output_file_path, gr.Button(value="Summarize", interactive=True)

    except Exception as e:
        log_messages.append(f"❌ An error occurred: {str(e)}")
        return f"An error occurred during summarization: {str(e)}", "\n".join(log_messages), None, gr.Button(value="Summarize", interactive=True)

# Gradio Interface
model_tip = gr.Markdown(
    """
    **Model Selection Tips:**
    - **facebook/bart-large-cnn:** Fast, general-purpose summarization for short to medium texts.
    - **google/long-t5-local-base:** Designed for long documents; better context handling.
    - **mistralai/Mistral-7B-v0.1:** High-quality nuanced summaries; resource-intensive.
    - **Custom Local Model:** Specify a path to a downloaded model (e.g., `./models/my-bart-model`).
    """
)

with gr.Blocks(css="#status-log { overflow-y: scroll; max-height: 200px; }") as demo:
    gr.Markdown("# Narrative Summarizer")
    gr.Markdown("Upload your text file OR enter text below to get a summarized version.")

    with gr.Row():
        file_input = gr.File(label="Upload Text File (.txt)", file_types=['.txt'])
        text_input = gr.Textbox(label="Or, paste your text here", lines=10)

    gr.Markdown("---")
    
    with gr.Accordion("Model & Prompt Settings", open=True):
        with gr.Row():
            model_dropdown = gr.Dropdown(choices=MODEL_OPTIONS, label="Choose Model", value=MODEL_OPTIONS[0])
            local_model_path_input = gr.Textbox(label="Local Model Path (optional)")
        
        model_tip.render()
        
        with gr.Row():
            prompt_dropdown = gr.Dropdown(choices=PROMPT_OPTIONS, label="Choose Prompt Type", value=PROMPT_OPTIONS[0])
            custom_prompt_input = gr.Textbox(label="Custom Prompt (use {chunk} placeholder)")

    gr.Markdown("---")

    with gr.Accordion("Advanced Parameters", open=False):
        with gr.Row():
            iterations_slider = gr.Slider(minimum=1, maximum=5, step=1, label="Summarization Iterations", value=1)
            batch_size_slider = gr.Slider(minimum=1, maximum=8, step=1, label="Batch Size (for GPU)", value=4)
        
        with gr.Row():
            target_word_count_slider = gr.Slider(minimum=20, maximum=200, step=10, label="Target Summary Word Count", value=50)

    summarize_button = gr.Button("Summarize")
    
    with gr.Row():
        output_text = gr.Textbox(label="Summary Output", lines=15)
        status_log = gr.Textbox(label="Process Log", lines=15, interactive=False, elem_id="status-log")
    
    download_button = gr.File(label="Download Summary", file_types=['.txt'])

    def update_ui_on_click():
        return gr.Button(interactive=False)

    summarize_button.click(
        fn=update_ui_on_click,
        outputs=summarize_button
    ).then(
        fn=run_app,
        inputs=[file_input, text_input, model_dropdown, local_model_path_input, prompt_dropdown, custom_prompt_input, iterations_slider, batch_size_slider, target_word_count_slider],
        outputs=[output_text, status_log, download_button, summarize_button]
    )

demo.launch()