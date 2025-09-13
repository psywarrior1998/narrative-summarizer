---
title: Narrative Summarizer
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.15.0
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

# 📚 Narrative Summarizer

Summarize long `.txt` narrative files into compressed, LLM-optimized summaries using BART. Choose between `Bread`, `Butter`, or both prompt styles for custom compression behavior. Upload a `.txt` file, select your preferences, and receive a clean, compressed summary in seconds.

---

## 📚 Narrative Summarizer — Hugging Face Space

**`psyrishi/narrative-summarizer`**

A user-friendly summarization tool for `.txt` files, powered by Hugging Face Transformers and built with Gradio.

This app transforms long-form narratives into compressed, LLM-friendly summaries using either the **"Bread"**, **"Butter"**, or a **combination of both** prompt styles. It supports checkpointing to avoid data loss on interruptions and ensures large text files are processed reliably.

---

### ✨ Features

* ✅ Supports `.txt` file uploads up to 3 MB (or more)
* 📌 Prompt options: `Bread`, `Butter`, or `Bread and Butter`
* 🔁 Multi-iteration summarization support
* 🧠 Model: `facebook/bart-large-cnn`
* 💾 Auto checkpointing: progress won't be lost on timeout
* 🧰 Output is saved for download post-processing
* 🌐 Clean Gradio UI – easy to run in browser

---

### 📥 How to Use

1. **Upload** a `.txt` file (max \~3MB recommended)
2. **Select** a summarization style from dropdown:

   * `Bread only`
   * `Butter only`
   * `Bread and Butter`
3. Choose:

   * `Iterations`: how many times the prompts apply
   * `Max Length`: max summary tokens per chunk
   * `Min Length`: min summary tokens per chunk
4. Click **Summarize**
5. Get your **condensed output** in the results box

---

### ⚙️ Tech Stack

| Component         | Details                          |
| ----------------- | -------------------------------- |
| **Frontend**      | [Gradio](https://www.gradio.app) |
| **Backend**       | Hugging Face `transformers`      |
| **Model**         | `facebook/bart-large-cnn`        |
| **Checkpointing** | JSON-based resume system         |
| **Language**      | Python 3.10+                     |

---

### 📂 Folder Structure

```
.
├── app.py              # Gradio frontend app
├── summarizer.py       # Backend summarization logic
├── requirements.txt    # Dependencies
├── inputs/             # Uploaded input files
├── outputs/            # Final summarized outputs
└── checkpoints/        # Intermediate checkpointing
```

---

### 🛠️ Setup (Local)

Clone this repo and run it locally:

```bash
git clone https://huggingface.co/spaces/psyrishi/narrative-summarizer
cd narrative-summarizer

pip install -r requirements.txt
python app.py
```

---

## 🚀 Space Configuration

Here’s how to fill out the **Hugging Face Space creation form**:

| Field                 | Value                                       |
| --------------------- | ------------------------------------------- |
| **Owner**             | `psyrishi`                                  |
| **Space Name**        | `narrative-summarizer`                      |
| **Short Description** | Summarizer for the txt files                |
| **License**           | Choose: `MIT`, `Apache 2.0`, or `Other`     |
| **Space SDK**         | ✅ Gradio                                    |
| **Gradio Template**   | Start from Scratch or Blank                 |
| **Hardware**          | ✅ Free (sufficient for your use case)       |
| **Visibility**        | Choose: `Public` (recommended) or `Private` |
| **Dev Mode**          | (Optional) Available to PRO subscribers     |

---

### 🧪 Prompt Styles Explained

* 🥖 **Bread**: Focuses on compression for efficient LLM parsing
* 🧈 **Butter**: Enhances nuance and detail while summarizing
* 🥪 **Bread + Butter**: Applies both sequentially for balance

---

### 📌 Example Input

```txt
Once upon a time, in a quiet village nestled between two mountains...
```

### 📤 Example Output (Bread only)

```txt
A peaceful mountain village faces hidden turmoil, gradually unveiling conflicts beneath its quiet surface.
```

---

### 🔐 License

Recommend using:

```
MIT License

Copyright (c) 2025 psyrishi
Permission is hereby granted, free of charge, to any person obtaining a copy...
```

Or [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

### 👋 Feedback & Contributions

Feel free to fork the repo, create pull requests, or open issues if you'd like to contribute or improve the tool.

