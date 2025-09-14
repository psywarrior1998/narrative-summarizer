import os
from transformers import pipeline
import chardet
import torch

class NarrativeSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        # Load the pipeline once and store it
        self.summarizer = pipeline("summarization", model=model_name, device=0 if torch.cuda.is_available() else -1)
        self.tokenizer = self.summarizer.tokenizer

    def chunk_text_tokenwise(self, text, max_tokens=512, overlap=50):
        """
        Splits text into token-based chunks with overlapping context.
        This is a superior method that prevents loss of information at chunk boundaries.
        """
        tokens = self.tokenizer.encode(text)
        chunks = []
        stride = max_tokens - overlap
        for i in range(0, len(tokens), stride):
            # Ensure the last chunk includes all remaining tokens
            end_index = min(i + max_tokens, len(tokens))
            chunk_tokens = tokens[i:end_index]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            if end_index == len(tokens):
                break
        return chunks

    def apply_custom_prompt(self, chunk, prompt_type):
        prompts = {
            "Bread only": "Transform the provided fictional narrative into a maximally compressed yet losslessly decompressible format optimized for LLM reconstruction. {chunk}",
            "Butter only": "Solid foundation, but let's refine the granularity. Your 4-subpoint structure creates artificial symmetry where organic complexity should flourish. {chunk}",
            "Bread and Butter": "Transform the provided fictional narrative into a maximally compressed format. Then refine granularity for organic complexity. {chunk}"
        }
        prompt_template = prompts.get(prompt_type, "{chunk}")
        return prompt_template.format(chunk=chunk)

    def summarize_chunk(self, chunk, prompt_type):
        prompt = self.apply_custom_prompt(chunk, prompt_type)
        summary = self.summarizer(prompt, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']

    def process_text(self, text, prompt_type, iterations=1):
        """
        Main summarization pipeline for a given text.
        Handles chunking, iterative passes, and final global compression.
        """
        if not text:
            return "No text provided to summarize."

        chunks = self.chunk_text_tokenwise(text, max_tokens=512, overlap=50)
        condensed_chunks = []

        # First pass summarization over all chunks
        for chunk in chunks:
            temp_chunk = chunk
            for _ in range(iterations):
                temp_chunk = self.summarize_chunk(temp_chunk, prompt_type)
            condensed_chunks.append(temp_chunk)

        # Second pass for global compression
        combined = " ".join(condensed_chunks)
        if len(combined.split()) > self.summarizer.tokenizer.model_max_length * 0.8:
            # Perform a final summary on the combined text only if it's large
            final_summary = self.summarize_chunk(combined, prompt_type)
        else:
            final_summary = combined

        return final_summary