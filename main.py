import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from flask import Flask, request, jsonify
from datasets import Dataset

# Set up environment for Google Colab or local setup
try:
    from google.colab import userdata
    os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
    os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
except ImportError:
    print("Running locally, ensure Kaggle credentials are set if needed.")

# Initialize Gemma 2B model and tokenizer
model_name = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Synthetic DSA dataset for fine-tuning
dataset = [
    {
        "prompt": "Problem: Given an array of integers, find the maximum subarray sum.\nInput: [-2,1,-3,4,-1,2,1,-5,4]\nOutput: 6\nWrite a Python solution.",
        "response": """def maxSubArray(nums):
    max_sum = nums[0]
    curr_sum = 0
    for num in nums:
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)
    return max_sum

# Explanation: This is Kadane's algorithm, which iterates through the array, keeping track of the maximum sum subarray ending at each position."""
    },
    {
        "prompt": "Problem: Reverse a linked list.\nInput: 1->2->3->4->5->NULL\nOutput: 5->4->3->2->1->NULL\nWrite a Python solution.",
        "response": """class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverseList(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev

# Explanation: This iterative solution reverses the linked list by changing the next pointer of each node to point to the previous node."""
    },
    {
        "prompt": "Problem: Find the two numbers in an array that sum to a target value.\nInput: nums = [2,7,11,15], target = 9\nOutput: [0,1]\nWrite a Python solution.",
        "response": """def twoSum(nums, target):
    num_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []

# Explanation: This solution uses a hash map to store numbers and their indices, achieving O(n) time complexity."""
    }
]

# Format dataset for fine-tuning
def format_dataset(dataset):
    formatted = [{"text": f"{item['prompt']}\n{item['response']}"} for item in dataset]
    return Dataset.from_list(formatted)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_dataset = format_dataset(dataset)
tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

# Configure LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=4,  # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Target attention layers
    lora_dropout=0.05
)
model = get_peft_model(model, lora_config)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./gemma_dsa_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=1e-5,
    fp16=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./gemma_dsa_finetuned")
tokenizer.save_pretrained("./gemma_dsa_finetuned")

# Initialize Flask app
app = Flask(__name__)

@app.route('/solve', methods=['POST'])
def solve_dsa():
    try:
        data = request.get_json()
        problem = data.get('problem', '')
        if not problem:
            return jsonify({"error": "No problem provided"}), 400
        prompt = f"Problem: {problem}\nWrite a Python solution."
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
        outputs = model.generate(**inputs, max_length=512)
        solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"solution": solution})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Example usage
def test_model():
    test_prompt = """Problem: Given a sorted array and a target value, find if there exists two elements that sum to the target.\nInput: nums = [2,7,11,15], target = 18\nOutput: True\nWrite a Python solution."""
    inputs = tokenizer(test_prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
    outputs = model.generate(**inputs, max_length=512)
    solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Test Problem Solution:")
    print(solution)

if __name__ == "__main__":
    # Run test in non-server mode
    test_model()
    # Start Flask app (comment out for Colab, use ngrok or local server)
    # app.run(host='0.0.0.0', port=5000)