
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class LLMDataGenerator:
    def __init__(self, model_name='gpt2', max_length=100, device=None):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.max_length = max_length

    def generate_text(self, prompt, num_samples=1):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=self.max_length,
            num_return_sequences=num_samples,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        return [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

    def generate_user_interactions(self, user_profile, num_items=5):
        prompt = f"User profile: {user_profile}\nRecommended items and interactions:"
        generations = self.generate_text(prompt, num_samples=1)
        return generations[0]

# Example usage
if __name__ == "__main__":
    generator = LLMDataGenerator()
    sample_profile = "User likes sci-fi books, rates above 4 stars for tech gadgets"
    print(generator.generate_user_interactions(sample_profile))
