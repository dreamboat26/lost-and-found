# LLama-7B-QLora-UltraChat

## Requirements
- peft==0.6.2
- transformers==4.35.2
- datasets==2.15.0
- bitsandbytes==0.41.2.post2
- trl==0.7.4
- accelerate==0.24.1
- wandb

## Model Description
The model is a 7B parameter Llama model that has been fine-tuned using LoRA and Quantization. LoRA is a method for efficient fine-tuning of large language models that involves adding low-rank decomposition matrices to the model's weights. Quantization is a technique for reducing the memory and computational requirements of the model by representing the weights using fewer bits.

## Fine-tuning
The model was fine-tuned on the UltraChat dataset using the SFTTrainer from the TRL library. The training arguments used are as follows:

- Output directory: `<YOUR_HF_USERNAME>/llama-7b-qlora-ultrachat`
- Per device train batch size: 4
- Gradient accumulation steps: 4
- Optimizer: Paged AdamW 32bit
- Save steps: 10
- Logging steps: 10
- Learning rate: 2e-4
- Max gradient norm: 0.3
- Max steps: 1000
- Warmup ratio: 0.03
- LR scheduler type: constant
- Gradient checkpointing: True

## Disclaimer

This model is for research purposes only and should not be used for any harmful or malicious purposes. The author is not responsible for any misuse of this model.

## Citation

If you use this model in your research, please cite the following papers:
- Hu, S., Liu, Z., Wei, L., & Li, X. (2021). Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.
- Roller, D., Kusner, M. J., & Ma, W. (2021). Benchmarking language models with human preferences. arXiv preprint arXiv:2102.07866.
- Touvron, H., de Masson d'Autume, E., Resende, P., Usunier, N., & Oliva, A. (2020). Training data-efficient image transformers & distillation through attention. 
    
## Usage
To use the model, you can load it using the following code:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "ybelkada/llama-7b-qlora-ultrachat"

tokenizer = AutoTokenizer.from_pretrained(model_id)

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    adapter_kwargs={"revision": "09487e6ffdcc75838b10b6138b6149c36183164e"}
)
You can then use the model to generate text as follows:
text = "### USER: Can you explain contrastive learning in machine learning in simple terms for someone new to the field of ML?### Assistant:"

inputs = tokenizer(text, return_tensors="pt").to(0)
outputs = model.generate(inputs.input_ids, max_new_tokens=250, do_sample=False)

print("After attaching Lora adapters:")
print(tokenizer.decode(outputs[0], skip_special_tokens=False))


