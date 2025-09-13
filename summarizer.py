import os
import json
import time
from transformers import pipeline
import torch

class Summarizer:
    def __init__(self, model_name="facebook/bart-large-cnn", chunk_size=1000, batch_size=4):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.device = 0 if torch.cuda.is_available() else -1
        self.summarizer = pipeline("summarization", model=model_name, device=self.device)

        os.makedirs("checkpoints", exist_ok=True)

    def _chunk_text(self, text):
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def _apply_prompt(self, chunk, prompt_type):
        if prompt_type == "Bread":
            return f"Transform the provided fictional narrative into a maximally compressed yet losslessly decompressible format optimized for LLM reconstruction. {chunk}"
        elif prompt_type == "Butter":
            return f"Solid foundation, but let's refine the granularity. Your 4-subpoint structure creates artificial symmetry where organic complexity should flourish. {chunk}"
        else:
            return chunk

    def summarize_file(self, input_path, output_path, prompt_types, iterations=1,
                       max_length=150, min_length=50, progress_callback=None):
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = self._chunk_text(text)
        total_chunks = len(chunks)
        processed_chunks = 0
        summaries = []
        start_time = time.time()

        # Checkpoint recovery
        checkpoint_path = os.path.join("checkpoints", os.path.basename(input_path) + ".json")
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r', encoding='utf-8') as cp:
                checkpoint_data = json.load(cp)
                summaries = checkpoint_data.get("summaries", [])
                processed_chunks = checkpoint_data.get("processed_chunks", 0)

        for i in range(processed_chunks, total_chunks):
            chunk = chunks[i]
            for _ in range(iterations):
                for p in prompt_types:
                    chunk = self._apply_prompt(chunk, p)

            summary = self.summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            summaries.append(summary)
            processed_chunks += 1

            # Save checkpoint
            with open(checkpoint_path, 'w', encoding='utf-8') as cp:
                json.dump({
                    "processed_chunks": processed_chunks,
                    "summaries": summaries
                }, cp)

            if progress_callback:
                elapsed = time.time() - start_time
                avg = elapsed / processed_chunks
                eta = avg * (total_chunks - processed_chunks)
                progress_callback(processed_chunks, total_chunks, eta)

        # Save final result
        final_summary = "\n".join(summaries)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_summary)

        return final_summary
