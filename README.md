---
title: Narrative Summarizer
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.45.0
app_file: app.py
pinned: false
license: mit
tags:
- summarization
- text
- transformer
- bart
- compression
- gradio
---

# Narrative Summarizer

A Gradio-based app to summarize large narrative `.txt` files using transformer models with advanced chunking and multi-pass summarization.

---

## Features

- **Robust UTF-8 file handling** with encoding detection for smooth uploads.
- **Token-based chunking** to handle large files efficiently.
- **Multiple prompt styles** via dropdown:
  - Bread Only
  - Butter Only
  - Bread and Butter
- **Iterative summarization passes** for better global compression.
- **Second-pass summarization** to refine and compress summaries further.
- Built with **Hugging Face Transformers** and **Gradio**.

---

## Setup

1. Clone this repo:
   ```bash
   git clone https://huggingface.co/spaces/psyrishi/narrative-summarizer
   cd narrative-summarizer
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app locally:

   ```bash
   python app.py
   ```

---

## Usage

* Upload a `.txt` file (UTF-8 or similar encodings supported).
* Choose a prompt style from the dropdown.
* Select the number of summarization iterations (≥1).
* Click **Summarize** to get the output.

---

## How It Works

* Reads the input file with encoding detection to avoid decode errors.
* Splits text into token-based chunks (\~200 tokens each).
* Applies custom prompts and summarizes each chunk.
* Optionally, performs multiple iterative passes to refine the summary.
* Combines chunk summaries and performs a second-pass summarization for global compression.

---

## Notes

* Model used: `facebook/bart-large-cnn` (can be customized in `summarizer.py`).
* GPU acceleration can speed up summarization if available.
* For very large files, increase iterations cautiously to avoid long runtimes.

---

## License

This project is licensed under the MIT License.

---

## Author

Created by [psyrishi](https://huggingface.co/psyrishi)

---

Feel free to contribute or raise issues!

