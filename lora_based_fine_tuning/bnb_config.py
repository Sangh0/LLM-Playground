import torch

from typing import Union, Optional
from transformers import BitsAndBytesConfig


def get_bnb_config(
    load_in_8bit: bool = False,
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: Optional[Union[torch.dtype, str]] = "bfloat16",
    bnb_4bit_use_double_quant: bool = False,
    bnb_4bit_quant_type: str = "nf4",
) -> BitsAndBytesConfig:
    """Bits and Bytes configuration for quantization.
    Args:
        load_in_8bit (`bool`): Whether to load the model in 8-bit precision.
        load_in_4bit (`bool`): Whether to load the model in 4-bit precision.
        bnb_4bit_compute_dtype (`Union[torch.dtype, str]`): The data type to use for 4-bit computation.
        bnb_4bit_use_double_quant (`bool`): Whether to use double quantization for 4-bit.
        bnb_4bit_quant_type (`str`): The quantization data type for 4-bit. Can be 'nf4' or 'fp4'.
    """
    if load_in_8bit and load_in_4bit:
        raise ValueError("Cannot load in both 8-bit and 4-bit.")

    if bnb_4bit_compute_dtype is None:
        bnb_4bit_compute_dtype = torch.float32
    elif isinstance(bnb_4bit_compute_dtype, str):
        bnb_4bit_compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    elif isinstance(bnb_4bit_compute_dtype, torch.dtype):
        bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
    else:
        raise ValueError("Invalid data type for bnb_4bit_compute_dtype.")

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
    )
    return bnb_config
