from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
local_dir = "local_deepseek_model"

# Download model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(local_dir)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True
)
model.save_pretrained(local_dir)

print("Model saved to:", local_dir)
