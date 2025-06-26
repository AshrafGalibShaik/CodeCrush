# 🧠 CodeCrush

> **AI-Powered Data Structures & Algorithms Problem Solver**  
> A fine-tuned Gemma 7B model optimized for coding interviews and competitive programming

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.2-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 What is DSA Assistant?

DSA Assistant is an intelligent coding companion that generates **optimal solutions** for Data Structures and Algorithms problems with detailed explanations. Built on Google's Gemma 7B model and fine-tuned with LoRA, it's your personal AI tutor for:

- **Coding Interview Preparation** (LeetCode, HackerRank)
- **Competitive Programming** (Codeforces, AtCoder)
- **Algorithm Learning** with complexity analysis
- **Code Optimization** and best practices

### ✨ Key Highlights

🎯 **Smart Problem Solving** - Handles arrays, trees, graphs, dynamic programming, and more  
⚡ **GPU Optimized** - Runs on budget GPUs like RTX 3050 (4GB VRAM)  
📊 **Complexity Analysis** - Provides time/space complexity for every solution  
🔧 **Extensible** - Easy to add new problems and expand capabilities  
🐍 **Python-First** - Generates clean, optimized Python code

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **AI Model** | Google Gemma 7B-it (instruction-tuned) |
| **Fine-tuning** | LoRA + PEFT for efficient training |
| **Optimization** | 4-bit quantization + DeepSpeed |
| **Framework** | PyTorch, Transformers, TRL |
| **Hardware** | NVIDIA RTX 3050+ (4GB VRAM minimum) |
| **Package Manager** | UV for fast dependency management |

---

## 🔧 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA RTX 3050 or better (4GB+ VRAM)
- **CPU**: AMD Ryzen 7 6800H or equivalent
- **RAM**: 16GB recommended
- **Storage**: 10GB free space

### Software Requirements
- **Python**: 3.8 or higher
- **CUDA**: 11.8 or 12.2 with compatible drivers
- **OS**: Windows 10/11 (64-bit) or Ubuntu 20.04+
- **UV**: Modern Python package manager

---

## ⚡ Quick Start

### 1. Clone & Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/dsa-assistant.git
cd dsa-assistant

# Create virtual environment with UV
uv venv
uv activate
```

### 2. Install Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Sync all dependencies
uv sync
```

### 3. Verify CUDA Installation
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
# Expected: CUDA Available: True, GPU: NVIDIA GeForce RTX 3050
```

### 4. Train the Model
```bash
# Start training (4-6 hours on RTX 3050)
uv run python dsa_assistant_starter_cuda.py

# Monitor GPU usage
nvidia-smi
```

### 5. Generate Solutions
```python
problem_prompt = """
### Problem:
Given an array of integers, return indices of the two numbers 
such that they add up to a target.

### Input: nums = [2,7,11,15], target = 9
### Solve in Python.
"""

solution = generate_dsa_solution(problem_prompt)
print(solution)
```

---

## 📁 Project Structure

```
dsa-assistant/
├── 🐍 dsa_assistant_starter_cuda.py    # Main training & inference script
├── 📋 dsa_problems.jsonl               # Curated dataset (20 problems)
├── ⚙️  ds_config.json                  # DeepSpeed configuration
├── 📦 pyproject.toml                   # UV dependency config
├── 📖 README.md                        # This documentation
└── 🤖 gemma_dsa_finetuned/             # Output model directory
```

---

## 🎯 Dataset Format

Add new problems to `dsa_problems.jsonl`:

```json
{
  "instruction": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
  "input": "nums = [2,7,11,15], target = 9",
  "output": "def twoSum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return [seen[complement], i]\n        seen[num] = i\n    return []\n\n# Time Complexity: O(n)\n# Space Complexity: O(n)"
}
```

---

## ⚙️ Configuration

### Training Parameters (RTX 3050 Optimized)

```python
training_args = TrainingArguments(
    output_dir="./gemma_dsa_finetuned",
    per_device_train_batch_size=1,        # Fit 4GB VRAM
    gradient_accumulation_steps=16,       # Effective batch size = 16
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,                           # Half precision for memory
    gradient_checkpointing=True,         # Trade compute for memory
    deepspeed="ds_config.json"           # CPU offloading
)
```

### Memory Optimization Features
- **4-bit Quantization**: Reduces model size by 75%
- **DeepSpeed ZeRO**: Offloads parameters to CPU/RAM
- **Gradient Checkpointing**: Lower memory usage during training
- **Dynamic Batching**: Adapts to available VRAM

---

## 🚨 Troubleshooting

### 🔥 CUDA Out of Memory
```python
# Reduce batch size in dsa_assistant_starter_cuda.py
per_device_train_batch_size=1
gradient_accumulation_steps=32  # Increase this

# Close other GPU applications
# Consider using Gemma 2B model for even lower VRAM usage
```

### 📦 Dependency Issues
```bash
# Force sync dependencies
uv sync --frozen

# Verify CUDA installation
python -c "import torch; print(torch.version.cuda)"
```

### 🐌 Slow Training
```bash
# Ensure fp16 is enabled
fp16=True

# Monitor GPU utilization
watch -n 1 nvidia-smi

# Consider smaller model variant
model_name = "google/gemma-2b-it"  # ~1.5GB VRAM
```

---

## 🎨 Usage Examples

### Two Sum Problem
```python
problem = """
### Problem: Two Sum
Given an array of integers nums and an integer target, 
return indices of two numbers that add up to target.

### Input: nums = [2,7,11,15], target = 9
### Solve in Python.
"""

solution = generate_dsa_solution(problem)
```

### Binary Tree Traversal
```python
problem = """
### Problem: Binary Tree Inorder Traversal
Given the root of a binary tree, return the inorder 
traversal of its nodes' values.

### Solve in Python.
"""

solution = generate_dsa_solution(problem)
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork** the repository
2. **Add** new DSA problems to `dsa_problems.jsonl`
3. **Test** your additions with the training script
4. **Submit** a pull request with clear descriptions

### Areas for Contribution
- More diverse problem types (graphs, dynamic programming)
- Multi-language support (Java, C++, JavaScript)
- Advanced optimization techniques
- Web interface for easy problem solving

---

## 📈 Performance Metrics

| Metric | RTX 3050 (4GB) | RTX 4060 (8GB) |
|--------|----------------|----------------|
| Training Time | 4-6 hours | 2-3 hours |
| Memory Usage | ~3.8GB VRAM | ~6.2GB VRAM |
| Batch Size | 1 | 2-4 |
| Problems/Hour | 3-5 | 6-10 |

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links & Resources

- **PyTorch**: [pytorch.org](https://pytorch.org/)
- **Transformers**: [huggingface.co/transformers](https://huggingface.co/transformers)
- **CUDA Toolkit**: [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- **UV Package Manager**: [github.com/astral-sh/uv](https://github.com/astral-sh/uv)

---

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/dsa-assistant/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/dsa-assistant/discussions)
- 📧 **Email**: your.email@example.com

---

<div align="center">

**Made with ❤️ for the coding community**

*Star ⭐ this repo if you find it helpful!*

</div>
