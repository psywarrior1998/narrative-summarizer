import os
from transformers import pipeline
import chardet
import torch

class NarrativeSummarizer:
    # Class-level cache to store pipelines
    _model_cache = {}

    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.model_name = model_name
        self._load_model()
        self.tokenizer = self.summarizer.tokenizer

    def _load_model(self):
        """Loads or retrieves the summarization pipeline from cache."""
        if self.model_name in self._model_cache:
            self.summarizer = self._model_cache[self.model_name]
        else:
            device = 0 if torch.cuda.is_available() else -1
            self.summarizer = pipeline(
                "summarization",
                model=self.model_name,
                device=device,
                # Setting max_length and min_length here ensures the pipeline is configured correctly
                # and avoids repeating these args for every call.
                max_length=150,
                min_length=50,
                do_sample=False
            )
            # Cache the model instance
            self._model_cache[self.model_name] = self.summarizer

    def chunk_text_tokenwise(self, text, max_tokens=512, overlap=50):
        """
        Splits text into token-based chunks with overlapping context.
        This is a superior method that prevents loss of information at chunk boundaries.
        """
        tokens = self.tokenizer.encode(text)
        chunks = []
        stride = max_tokens - overlap
        for i in range(0, len(tokens), stride):
            end_index = min(i + max_tokens, len(tokens))
            chunk_tokens = tokens[i:end_index]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            if end_index == len(tokens):
                break
        return chunks

    def apply_custom_prompt(self, chunk, prompt_type, custom_prompt):
        """Applies a custom prompt template to a text chunk."""
        if custom_prompt:
            return custom_prompt.format(chunk=chunk)

        prompts = {
            "Bread only": "Transform the provided fictional narrative into a maximally compressed yet losslessly decompressible format optimized for LLM reconstruction. {chunk}",
            "Butter only": "Solid foundation, but let's refine the granularity. Your 4-subpoint structure creates artificial symmetry where organic complexity should flourish. {chunk}",
            "Bread and Butter": "Transform the provided fictional narrative into a maximally compressed format. Then refine granularity for organic complexity. {chunk}"
        }
        prompt_template = prompts.get(prompt_type, "{chunk}")
        return prompt_template.format(chunk=chunk)

    def summarize_batch(self, chunks, prompt_type, custom_prompt, target_summary_word_count=50):
        """
        Summarizes a batch of text chunks.
        This method leverages model parallelization for increased efficiency.
        """
        processed_chunks = [self.apply_custom_prompt(chunk, prompt_type, custom_prompt) for chunk in chunks]
        
        # Determine max_length based on user-provided word count
        # A simple approximation: 1.3 tokens per word
        target_tokens = int(target_summary_word_count * 1.3)
        
        summaries = self.summarizer(
            processed_chunks,
            max_length=target_tokens + 20, # Add a small buffer
            min_length=target_tokens - 10,
            do_sample=False,
            # For better performance on CPU, you can use fp32, but for GPU, fp16 is better
            # Note: This is an advanced optimization and may require specific hardware/software setup
            # torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        return [s['summary_text'] for s in summaries]

    def process_text(self, text, prompt_type, custom_prompt, iterations=1, batch_size=4, target_word_count=50):
        """
        Main summarization pipeline for a given text.
        Handles chunking, iterative passes, and final global compression.
        """
        if not text:
            return "No text provided to summarize."
        
        chunks = self.chunk_text_tokenwise(text, max_tokens=512, overlap=50)
        condensed_chunks = []
        
        # First pass summarization over all chunks
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            for _ in range(iterations):
                # The model will run a batch of summaries
                batch_summaries = self.summarize_batch(batch, prompt_type, custom_prompt, target_word_count)
                # Update the chunks for the next iteration with the new summaries
                batch = batch_summaries
            condensed_chunks.extend(batch)

        # Second pass for global compression
        combined = " ".join(condensed_chunks)
        
        # Check if the combined text is large enough for a final summary
        if len(self.tokenizer.encode(combined)) > self.tokenizer.model_max_length * 0.8:
            final_summary = self.summarize_batch([combined], prompt_type, custom_prompt, target_word_count)[0]
        else:
            final_summary = combined

        return final_summary