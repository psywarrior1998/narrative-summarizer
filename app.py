import gradio as gr
from summarizer import NarrativeSummarizer

# Initialize summarizer instance (can specify model etc here)
summarizer = NarrativeSummarizer()

def run_summarization(file, prompt_type, iterations):
    if not file:
        return "❌ Error: No file uploaded."
    try:
        iterations = int(iterations)
        if iterations < 1:
            return "❌ Error: Iterations must be >= 1."
    except ValueError:
        return "❌ Error: Iterations must be an integer."
    
    try:
        # Run summarization
        summary = summarizer.process_file(file.name, prompt_type, iterations)
        return summary
    except Exception as e:
        return f"❌ Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# Narrative Summarizer")
    with gr.Row():
        file_input = gr.File(label="Upload your .txt file")
        prompt_dropdown = gr.Dropdown(
            choices=["Bread Only", "Butter Only", "Bread and Butter"],
            value="Bread Only",
            label="Select Prompt Type"
        )
        iterations_input = gr.Number(value=1, label="Iterations", precision=0, minimum=1)

    output_text = gr.Textbox(label="Summary Output", lines=15)

    run_button = gr.Button("Summarize")
    run_button.click(
        fn=run_summarization,
        inputs=[file_input, prompt_dropdown, iterations_input],
        outputs=output_text
    )

demo.launch()
