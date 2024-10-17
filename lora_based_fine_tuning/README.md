## LoRA-based Fine-tuning for LLMs

### Features

1. **LLM Fine-Tuning with LoRA, DoRA, and AdaLoRA**:
   - Fine-tune large language models using parameter-efficient methods such as [`LoRA`](https://arxiv.org/abs/2106.09685), [`DoRA`](https://arxiv.org/abs/2402.09353), and [`AdaLoRA`](https://arxiv.org/abs/2303.10512).

2. **Quantization with `bitsandbytes`**:
   - Supports model quantization during training to reduce memory consumption and accelerate inference.

3. **Fine-Tuning Acceleration with `unsloth`**:
   - Accelerates the fine-tuning process, providing optimized model loading and training routines for faster results.

### Installation

#### Poetry Setup
To install and set up the project using Poetry, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/Sangh0/LLM-Playground.git
cd LLM-Playground/lora_based_fine_tuning
```

2. Activate the virtual environment:
```bash
poetry shell
```

3. Install dependencies:
```bash
poetry install
```

#### Docker Setup
To run the project in a Docker container:

1. Build the Docker image:
```bash
docker build -t lora-fine-tuning .
```

2. Run the container:
```bash
docker run --gpus all -it lora-fine-tuning
```

#### Pip Setup (Alternative)
If you prefer to use `pip` and a `requirements.txt`:
```bash
pip install -r requirements.txt
```
