import gradio as gr
from summarizer import load_model, summarize_chunks

model_options = [
    "facebook/bart-large-cnn",
    "google/pegasus-xsum",
    "allenai/led-base-16384",
    "psyrishi/llama2-7b-summary"
]

def summarize_file(file_obj, compression_level, model_name):
    try:
        text = file_obj.read().decode("utf-8")
    except:
        return "❌ Error: Unable to read the file. Please upload a valid UTF-8 text file."

    summarizer, tokenizer = load_model(model_name)
    result = summarize_chunks(text, summarizer, tokenizer, compression_level=compression_level, second_pass=True)
    return result

with gr.Blocks() as demo:
    gr.Markdown("## 📚 Advanced Narrative Summarizer")
    gr.Markdown("Summarize large `.txt` files using advanced transformers like Longformer, LLaMA2, and Pegasus.")

    with gr.Row():
        file_input = gr.File(label="Upload .txt File", file_types=[".txt"])
        compression_dropdown = gr.Dropdown(
            choices=[
                "High (90% compression)",
                "Medium (70% compression)",
                "Low (50% compression)"
            ],
            value="Medium (70% compression)",
            label="Compression Level"
        )
        model_dropdown = gr.Dropdown(choices=model_options, value=model_options[0], label="Model")

    summarize_btn = gr.Button("Summarize")

    output_text = gr.Textbox(label="📄 Summarized Output", lines=20, interactive=False)

    summarize_btn.click(fn=summarize_file, inputs=[file_input, compression_dropdown, model_dropdown], outputs=output_text)

demo.launch(share=True)
