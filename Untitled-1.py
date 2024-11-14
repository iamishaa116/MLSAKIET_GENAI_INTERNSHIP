from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load a pre-trained model and tokenizer for summarization
model_name = "facebook/bart-large-cnn"  # You can also try "t5-small", "t5-base", "t5-large", etc.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Function to summarize text
def summarize_text(text, max_length=150, min_length=40, length_penalty=2.0):
    # Tokenize input text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate summary
    with torch.no_grad():
        summary_ids = model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=length_penalty,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode and return the summary
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Example usage
text = """
    The quick brown fox jumps over the lazy dog. In a distant land, the fox met many other animals and explored a forest full of mysteries. 
    This forest had tall trees, rivers, and a vast variety of plants and creatures. The fox quickly made friends with a squirrel, a rabbit, 
    and even a deer. Together, they faced challenges, solved mysteries, and enjoyed their time in the magical forest. Every day was an adventure, 
    and they learned valuable lessons from each other. Finally, the fox decided to return home, filled with memories and stories to share.
"""
summary = summarize_text(text)
print("Summary:", summary)
