import torch

from typing import Optional, Union, List, Dict
from unsloth import FastLanguageModel


def get_unsloth_model_and_tokenizer(
    model_name: str,
    max_seq_length: int,
    dtype: Optional[Union[str, torch.dtype]] = None,
    load_in_4bit: bool = False,
    device_map: Dict = {"": 0},
    trust_remote_mode: bool = False,
):
    """Use the unsloth framework to load a model and tokenizer from the Hugging Face model hub.
    Args:
        model_name (`str`): The name of the model to load.
        max_seq_length (`int`): The maximum sequence length for the model.
        dtype (`Optional[Union[str, torch.dtype]]`): The data type for the model.
        load_in_4bit (`bool`): Whether to load the model in 4-bit mode.
        device_map (`Dict`): The device map for tensor parallelism.
        trust_remote_mode (`bool`): Whether to trust the remote mode for the model.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        device_map=device_map,
        trust_remote_mode=trust_remote_mode,
    )

    return model, tokenizer


def get_unsloth_lora_model(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    bias: str = "none",
    target_modules: Union[List[str], str] = ["q_proj", "v_proj"],
):
    """Use the unsloth framework to load a LoRA adapter.
    Args:
        model: The model to load used unsloth.
        lora_r: The number of tensor parallelism.
        lora_alpha: The number of bits for quantization.
        lora_dropout: The dropout rate for LoRA.
        bias: The bias for LoRA.
        target_modules: The target modules for Lo
    """
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
    )
    return model
