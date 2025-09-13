from transformers import pipeline, AutoTokenizer
import math

def load_model(model_name="facebook/bart-large-cnn"):
    summarizer = pipeline("summarization", model=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return summarizer, tokenizer

def chunk_text_by_tokens(text, tokenizer, max_tokens=1024):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

def get_summary_lengths(token_count, compression_level):
    if compression_level == "High (90% compression)":
        factor = 0.1
    elif compression_level == "Medium (70% compression)":
        factor = 0.3
    else:
        factor = 0.5

    max_len = max(30, math.ceil(token_count * factor))
    min_len = max(10, math.ceil(token_count * (factor / 2)))
    return min_len, max_len

def summarize_chunks(text, summarizer, tokenizer, compression_level="Medium (70% compression)", second_pass=True):
    chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=1024)
    summaries = []

    for chunk in chunks:
        token_count = len(tokenizer.encode(chunk))
        min_len, max_len = get_summary_lengths(token_count, compression_level)
        try:
            summary = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
        except Exception as e:
            summary = f"[Error summarizing chunk: {e}]"
        summaries.append(summary.strip())

    combined_summary = "\n\n".join(summaries)

    # Second-pass summarization (global)
    if second_pass and len(summaries) > 1:
        token_count = len(tokenizer.encode(combined_summary))
        min_len, max_len = get_summary_lengths(token_count, compression_level)
        try:
            final_summary = summarizer(combined_summary, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
            return final_summary
        except Exception as e:
            return combined_summary + f"\n\n[Second pass error: {e}]"
    else:
        return combined_summary
