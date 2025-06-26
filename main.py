import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

# Step 1: Load Gemma 7B-it model
model_name = "google/gemma-7b-it"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 2: Define LoRA configuration for fine-tuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Step 3: Load and format DSA dataset (example JSONL format)
dataset = load_dataset("json", data_files="dsa_problems.jsonl")

# Step 4: Define training arguments
training_args = TrainingArguments(
    output_dir="./gemma_dsa_finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    push_to_hub=False,
)

# Step 5: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

# Step 6: Fine-tune the model
trainer.train()

# Step 7: Inference function for generating solutions
def generate_dsa_solution(prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    problem_prompt = """
    ### Problem:
    Given an array of integers, return indices of the two numbers such that they add up to a target.
    ### Solve in Python.
    """
    solution = generate_dsa_solution(problem_prompt)
    print("Generated Solution:\n", solution)

# Optional: Save the fine-tuned model
model.save_pretrained("./gemma_dsa_finetuned")
tokenizer.save_pretrained("./gemma_dsa_finetuned")