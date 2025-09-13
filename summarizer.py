import os
from transformers import pipeline
import chardet

class NarrativeSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn", chunk_size=1000):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.summarizer = pipeline("summarization", model=self.model_name)
    
    def chunk_text_token_based(self, text):
        # Token-based chunking approximation based on whitespace split (could be improved with tokenizers)
        words = text.split()
        chunks = []
        current_chunk = []
        current_len = 0
        max_tokens = 200  # approximate token limit per chunk (adjust as needed)
        for word in words:
            current_chunk.append(word)
            current_len += 1
            if current_len >= max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def apply_custom_prompt(self, chunk, prompt_type):
        if prompt_type == "Bread Only":
            prompt = f"Transform the provided fictional narrative into a maximally compressed yet losslessly decompressible format optimized for LLM reconstruction. {chunk}"
        elif prompt_type == "Butter Only":
            prompt = f"Solid foundation, but let's refine the granularity. Your 4-subpoint structure creates artificial symmetry where organic complexity should flourish. {chunk}"
        elif prompt_type == "Bread and Butter":
            prompt = f"Transform the provided fictional narrative into a maximally compressed format. Then refine granularity for organic complexity. {chunk}"
        else:
            prompt = chunk
        return prompt

    def summarize_chunk(self, chunk, prompt_type):
        prompt = self.apply_custom_prompt(chunk, prompt_type)
        summary = self.summarizer(prompt, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']

    def process_file(self, file_path, prompt_type, iterations=1):
        # Read file robustly with encoding detection
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
            detected = chardet.detect(raw_data)
            encoding = detected['encoding'] or 'utf-8'
            text = raw_data.decode(encoding, errors='replace')
        except Exception as e:
            raise RuntimeError(f"Unable to read the file: {str(e)}")

        # Chunk the text token-wise
        chunks = self.chunk_text_token_based(text)
        condensed_chunks = []

        for chunk in chunks:
            temp_chunk = chunk
            for _ in range(iterations):
                temp_chunk = self.apply_custom_prompt(temp_chunk, prompt_type)
                temp_chunk = self.summarize_chunk(temp_chunk, prompt_type)
            condensed_chunks.append(temp_chunk)

        # Second pass summarization for global compression
        combined = " ".join(condensed_chunks)
        if iterations > 1:
            final_summary = self.summarize_chunk(combined, prompt_type)
        else:
            final_summary = combined

        return final_summary
