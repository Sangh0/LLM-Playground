from typing import Union, List
from peft import LoraConfig, AdaLoraConfig


def get_lora_config(
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    bias: str = "none",
    target_modules: Union[List[str], str] = ["q_proj", "v_proj"],
    task_type: str = "CAUSAL_LM",
) -> LoraConfig:
    """Load a LoRA configuration.
    Args:
        lora_r (`int`): The rank of the low-rank approximation.
        lora_alpha (`int`): The alpha parameter for LoRA scaling.
        lora_dropout (`float`): The dropout rate for LoRA layers.
        bias (`str`): The bias type for LoRA. Can be 'none', 'all' or 'lora_only'.
        target_modules (`Union[List[str], str]`): The target modules to apply LoRA.
        task_type (`str`): The task type for the model.
    """
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
        target_modules=target_modules,
    )
    return lora_config


def get_adalora_config(
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    init_r: int = 16,
    target_r: int = 4,
    tinit: int = 100,
    tfinal: int = 500,
    deltaT: int = 10,
    beta1: float = 0.85,
    beta2: float = 0.85,
    orth_reg_weight: float = 0.5,
    target_modules: Union[List[str], str] = [
        "k_proj",
        "q_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
    ],
    task_type: str = "CAUSAL_LM",
) -> AdaLoraConfig:
    """Load an AdaLoRA configuration.
    Args:
        lora_alpha (`int`): The alpha parameter for LoRA scaling.
        lora_dropout (`float`): The dropout rate for LoRA layers.
        target_r (`int`): The target rank for AdaLoRA.
        tinit (`int`): The initial temperature for AdaLoRA.
        init_r (`int`): The initial rank for AdaLoRA.
        tfinal (`int`): The final temperature for AdaLoRA.
        deltaT (`int`): The temperature decay step for AdaLoRA.
        beta1 (`float`): The beta1 parameter of EMA for sensitivity smoothing.
        beta2 (`float`): The beta2 parameter of EMA for undertainty quantification.
        orth_reg_weight (`float`): The weight of the orthogonality regularization term.
        target_modules (`Union[List[str], str]`): The target modules to apply AdaLoRA.
        task_type (`str`): The task type for the model.
    """
    adalora_config = AdaLoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        init_r=init_r,
        target_r=target_r,
        tinit=tinit,
        tfinal=tfinal,
        deltaT=deltaT,
        beta1=beta1,
        beta2=beta2,
        orth_reg_weight=orth_reg_weight,
        target_modules=target_modules,
        task_type=task_type,
    )
    return adalora_config


def get_dora_config(
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    bias: str = "none",
    target_modules: Union[List[str], str] = ["q_proj", "v_proj"],
    task_type: str = "CAUSAL_LM",
) -> LoraConfig:
    """Load a DoRA configuration."""
    lora_config = get_lora_config(
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        target_modules=target_modules,
        task_type=task_type,
    )
    lora_config.use_dora = True
    return lora_config
