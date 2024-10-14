from typing import Optional, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_model_and_tokenizer(
    model_name: str,
    bnb_config: Optional[BitsAndBytesConfig] = None,
    use_cache: bool = False,
    pretraining_tp: int = 1,
    device_map: Dict = {"": 0},
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model and tokenizer from the Hugging Face model hub.
    Args:
        model_name (`str`): The name of the model to load.
        bnb_config (`Optional[BitsAndBytesConfig]`): The Bits and Bytes configuration for quantization.
        use_cache (`bool`): Whether to use the cache for the model.
        pretraining_tp (`int`): The number of tensor parallelism, if set to 1, active only one GPU device.
        device_map (`Dict`): The device map for tensor parallelism.
    """

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
    )
    model.config.use_cache = use_cache
    model.config.pretraining_tp = pretraining_tp

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer
