import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# -------------------------------
# Replacement for top_k_top_p_filtering
# -------------------------------
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
	""" Filter a distribution of logits using top-k and/or nucleus (top-p) filtering """
	top_k = min(top_k, logits.size(-1))
	if top_k > 0:
		indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
		logits[indices_to_remove] = filter_value

	if top_p > 0.0:
		sorted_logits, sorted_indices = torch.sort(logits, descending=True)
		cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

		sorted_indices_to_remove = cumulative_probs > top_p
		sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
		sorted_indices_to_remove[..., 0] = 0

		indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
		logits[indices_to_remove] = filter_value

	return logits

# -------------------------------
# Oxygen + Heat Transformer Wrapper
# -------------------------------
class OxygenHeatWrapper(nn.Module):
	def __init__(self, model, oxygen_level=0.3, heat=0.2, vacuum=False):
		super().__init__()
		self.model = model
		self.oxygen_level = oxygen_level
		self.heat = heat
		self.vacuum = vacuum

	def forward(self, input_ids):
		outputs = self.model.transformer(input_ids, output_hidden_states=True)
		hidden_states = list(outputs.hidden_states)

		new_hidden_states = []
		for h in hidden_states:
			if self.vacuum:
				h = h + self.heat * torch.tanh(h)  # Only heat
			else:
				diffusion = self.oxygen_level * (torch.roll(h, shifts=1, dims=1) - h)
				h = h + diffusion + self.heat * torch.tanh(h)
			new_hidden_states.append(h)

		last_h = new_hidden_states[-1]
		logits = self.model.lm_head(last_h)
		return logits

# -------------------------------
# Sampling function
# -------------------------------
def sample_logits(logits, temperature=1.0, top_k=50, top_p=0.9):
	logits = logits / temperature
	filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
	probs = F.softmax(filtered_logits, dim=-1)
	next_token = torch.multinomial(probs, num_samples=1)
	return next_token

# -------------------------------
# Interactive Chat
# -------------------------------
def chat():
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

	print("=== Oxygen + Heat Chat Demo ===")
	print("Type 'vacuum' to toggle vacuum mode, 'exit' to quit.\n")

	oxygen_level = 0.3
	heat = 0.2
	vacuum = False

	wrapper = OxygenHeatWrapper(model, oxygen_level, heat, vacuum)

	while True:
		prompt = input("You: ")
		if prompt.lower() == "exit":
			break
		if prompt.lower() == "vacuum":
			vacuum = not vacuum
			wrapper.vacuum = vacuum
			print(f"[Mode switched to {'vacuum' if vacuum else 'oxygen + heat'}]")
			continue

		input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
		generated_ids = input_ids.clone()
		max_length = 30  # max generated tokens

		for _ in range(max_length):
			logits = wrapper(generated_ids)
			next_token = sample_logits(logits[:, -1, :], temperature=1.0)
			generated_ids = torch.cat([generated_ids, next_token], dim=1)

		output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
		print(f"Bot: {output_text}\n")

# -------------------------------
# Run Chat
# -------------------------------
if __name__ == "__main__":
	chat()
