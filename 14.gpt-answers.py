import torch
import torch.nn as nn
from torchtyping import TensorType

#the model is a GPT model trained to generate some type of text, you may want to change the model to a different type of model
class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        generated_tokens = []
        for i in range(new_chars):
            # Get model predictions for the current context.
            logits = model(context)  # shape: [1, time, vocab_size]
            logits = logits[:, -1, :]  # only the last timestep: shape [1, vocab_size]
            probs = torch.softmax(logits, dim=-1)  # convert scores to probabilities
            
            # Restore generator state for reproducibility and sample next token.
            generator.set_state(initial_state)
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)  # shape: [1, 1]
            
            # Append the sampled token to our context.
            context = torch.cat([context, next_token], dim=1)
            if context.shape[1] > context_length:
                context = context[:, -context_length:]
            generated_tokens.append(next_token.item())
        
        # Convert generated tokens to their character representations and join to form output.
        return ''.join([int_to_char[t] for t in generated_tokens])
