# ERNIE-4.5 Fine-tuned for Mathematical Reasoning

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/naazimsnh02/ernie-45-math-finetuned)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A fine-tuned version of ERNIE-4.5-21B specialized in solving complex mathematical problems using QLoRA on the Nemotron-RL Math dataset.

## Overview

This project demonstrates fine-tuning the ERNIE-4.5-21B model for mathematical reasoning tasks including algebra, calculus, geometry, and competition-level mathematics. The model was trained using efficient QLoRA (4-bit quantization + LoRA) on Modal's serverless GPU infrastructure.

**Model Card**: [naazimsnh02/ernie-45-math-finetuned](https://huggingface.co/naazimsnh02/ernie-45-math-finetuned)

## Key Features

- **Base Model**: ERNIE-4.5-21B (21 billion parameters)
- **Training Method**: QLoRA (4-bit quantization + LoRA adapters)
- **Dataset**: NVIDIA Nemotron-RL-math-OpenMathReasoning (8,000 samples)
- **Training Infrastructure**: Modal serverless GPU (40GB A100)
- **Optimization**: Unsloth framework (2x faster, 70% less memory)
- **Trainable Parameters**: ~0.15% of total (highly efficient)

## Performance

| Metric | Value |
|--------|-------|
| Final Training Loss | 0.604 |
| Best Validation Loss | 0.611 |
| Training Steps | 700 |
| Training Time | ~4.6 hours |
| Peak GPU Memory | 37.5 GB / 40 GB |
| Loss Improvement | 9.2% |

## Installation

```bash
# Clone the repository
git clone https://github.com/naazimsnh02/ernie-45-math-finetuned.git
cd ernie-45-math-finetuned

# Install dependencies
pip install unsloth[cu128-torch270]==2025.7.8
pip install transformers==4.56.2
pip install datasets==3.6.0
pip install trl==0.22.2
```

## Quick Start

### Using the Fine-tuned Model

```python
from unsloth import FastModel

# Load the fine-tuned model
model, tokenizer = FastModel.from_pretrained(
    model_name="naazimsnh02/ernie-45-math-finetuned",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Prepare for inference
FastModel.for_inference(model)

# Solve a math problem
messages = [{
    "role": "user",
    "content": "Solve the equation: 2x² + 5x - 3 = 0"
}]

prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Training Your Own Model

The complete training pipeline is available in the Jupyter notebook:

```bash
jupyter notebook ernie-45-fine-tuned-for-mathematical-reasoning.ipynb
```

Or run on Modal (recommended for GPU access):

```bash
modal run ernie-45-fine-tuned-for-mathematical-reasoning.ipynb
```

## Training Configuration

### Model Architecture
- **Base**: unsloth/ERNIE-4.5-21B-A3B-PT
- **LoRA Rank**: 16
- **LoRA Alpha**: 16
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Hyperparameters
- **Batch Size**: 4 (per device)
- **Gradient Accumulation**: 2 steps
- **Effective Batch Size**: 8
- **Learning Rate**: 2e-4
- **LR Scheduler**: Cosine with 5% warmup
- **Optimizer**: AdamW 8-bit
- **Precision**: BF16
- **Max Sequence Length**: 2048 tokens

### Dataset
- **Source**: nvidia/Nemotron-RL-math-OpenMathReasoning
- **Training Samples**: 7,600
- **Validation Samples**: 400
- **Split Ratio**: 95% train / 5% eval

## Example Outputs

**Problem**: Solve the equation: x² + 5x + 6 = 0

**Model Output**:
```
To solve x² + 5x + 6 = 0, we can factor:

Find two numbers that multiply to 6 and add to 5:
2 and 3 work because 2 × 3 = 6 and 2 + 3 = 5

Factored form:
(x + 2)(x + 3) = 0

Setting each factor to zero:
x + 2 = 0  →  x = -2
x + 3 = 0  →  x = -3

Therefore: \boxed{x = -2, -3}
```

## Project Structure

```
ernie-45-math-finetuned/
├── ernie-45-fine-tuned-for-mathematical-reasoning.ipynb  # Main training notebook
├── README.md                                              # This file
└── requirements.txt                                       # Python dependencies
```

## Training Progress

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 100  | 0.589         | 0.673          |
| 200  | 0.661         | 0.648          |
| 300  | 0.637         | 0.646          |
| 400  | 0.557         | 0.640          |
| 500  | 0.587         | 0.633          |
| 600  | 0.589         | 0.617          |
| 700  | 0.605         | 0.611          |

Training was stopped at step 700 for optimal validation performance.

## Requirements

- Python 3.8+
- CUDA-capable GPU (minimum 24GB VRAM recommended)
- PyTorch 2.7.0+
- Transformers 4.56.2
- Unsloth 2025.7.8
- Modal account (for cloud training)

## Use Cases

This model excels at:
- Solving algebraic equations and inequalities
- Factoring polynomials
- Calculus problems (derivatives, integrals)
- Geometry and trigonometry
- Word problems requiring multi-step reasoning
- Competition-level mathematics

## Limitations

- Optimized for mathematical reasoning; may not perform as well on other domains
- Trained on English language problems only
- Best results with problems similar to training data format
- Requires GPU for inference (4-bit quantization)

## Citation

```bibtex
@misc{ernie45-math-2025,
  title={ERNIE-4.5 Fine-tuned for Mathematical Reasoning},
  author={naazimsnh02},
  year={2025},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/naazimsnh02/ernie-45-math-finetuned}}
}
```

## Acknowledgments

- **ERNIE Team** for the base model
- **Unsloth** ([unslothai](https://github.com/unslothai/unsloth)) for the optimization framework
- **NVIDIA** for the Nemotron-RL dataset
- **Modal** ([modal.com](https://modal.com)) for GPU infrastructure
- **ERNIE AI Developer Challenge** for the opportunity

## License

MIT License - See [LICENSE](LICENSE) file for details

## Contact

For questions or feedback:
- HuggingFace: [@naazimsnh02](https://huggingface.co/naazimsnh02)
- Model Issues: [Open an issue](https://github.com/naazimsnh02/ernie-45-math-finetuned/issues)

---

**Trained with ❤️ using Unsloth and Modal**
