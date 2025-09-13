import os
import gradio as gr
from summarizer import Summarizer

summarizer = Summarizer()

PROMPT_CHOICES = {
    "Bread only": ["Bread"],
    "Butter only": ["Butter"],
    "Bread and Butter": ["Bread", "Butter"]
}

def summarize_file(file, prompt_type, iterations, max_length, min_length):
    if not file:
        return "No file uploaded."

    os.makedirs("inputs", exist_ok=True)
    input_path = os.path.join("inputs", file.name)
    with open(input_path, 'wb') as f:
        f.write(file.read())

    output_path = os.path.join("outputs", f"{os.path.splitext(file.name)[0]}_summary.txt")
    os.makedirs("outputs", exist_ok=True)

    def progress_callback(done, total, eta):
        return print(f"Progress: {done}/{total} | ETA: {int(eta)} sec")

    try:
        summary = summarizer.summarize_file(
            input_path=input_path,
            output_path=output_path,
            prompt_types=PROMPT_CHOICES[prompt_type],
            iterations=iterations,
            max_length=max_length,
            min_length=min_length,
            progress_callback=progress_callback
        )
        return summary
    except Exception as e:
        return f"Error occurred during summarization: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("## 📚 Narrative Compression Tool")

    with gr.Row():
        file_input = gr.File(label="Upload .txt File", file_types=[".txt"])
        prompt_type = gr.Dropdown(
            choices=list(PROMPT_CHOICES.keys()),
            label="Select Prompt",
            value="Bread only"
        )

    iterations = gr.Slider(1, 5, value=1, step=1, label="Iterations")
    max_length = gr.Slider(50, 300, value=150, step=10, label="Max Summary Length")
    min_length = gr.Slider(20, 100, value=50, step=10, label="Min Summary Length")

    submit = gr.Button("Summarize")

    output = gr.Textbox(label="Condensed Summary", lines=15)

    submit.click(
        summarize_file,
        inputs=[file_input, prompt_type, iterations, max_length, min_length],
        outputs=output
    )

demo.launch()
