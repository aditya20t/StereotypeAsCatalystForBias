import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig

def quantize():
    # QLoRA Config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True, # enable 4-bit quantization
        bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
        bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
        bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
    )


    # LORA Config
    lora_config = LoraConfig(
        r = 16, # Rank
        lora_alpha=8, # Scaling factor
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], # Modules to apply LORA
        lora_dropout=0.01, # Dropout
        bias='none', # Bias
    )

    return quantization_config, lora_config