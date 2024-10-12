from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Union
from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer


def get_dataset(
    dataset_name: str, split: str = "train"
) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    """Load a dataset from the Hugging Face datasets library.
    Args:
        dataset_name (`str`): The name of the dataset to load.
        split (`str`): The split of the dataset to load. Must be one of "train" or "validation".
    """
    assert split in ("train", "validation")
    return load_dataset(dataset_name, split=split)


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


@dataclass
class TrainingArgs:
    """Training arguments for the model training loop.
    Args:
        output_dir (`str`): The output directory to save the model and logs.
        eval_strategy (`str`): The evaluation strategy to use. Can be 'epoch' or 'steps'.
        per_device_train_batch_size (`int`): The batch size per GPU for training.
        per_device_eval_batch_size (`int`): The batch size per GPU for evaluation.
        gradient_accumulation_steps (`int`): The number of gradient accumulation steps.
        learning_rate (`float`): The learning rate for training.
        weight_decay (`float`): The weight decay for training.
        max_grad_norm (`float`): The maximum gradient norm for clipping.
        epochs (`int`): The number of epochs to train for.
        lr_scheduler_type (`str`): The learning rate scheduler to use.
        warmup_ratio (`float`): The warmup ratio for the learning rate scheduler.
        logging_strategy (`str`): The logging strategy to use. Can be 'epoch' or 'steps'.
        save_total_limit (`int`): The maximum number of checkpoints to save.
        bf16 (`bool`): Whether to use bfloat16 precision.
        fp16 (`bool`): Whether to use mixed precision training.
        load_best_model_at_end (`bool`): Whether to load the best model at the end of training.
        optim (`str`): The optimizer to use. Can be 'adamw_torch' or 'adamw'.
        report_to (`str`): The logging service to report to. Can be 'tensorboard' or 'wandb'.
    """

    output_dir: str = "./output"
    eval_strategy: str = "epoch"
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 1
    lr: float = 5e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 3.0
    epochs: int = 10
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    logging_strategy: str = "epoch"
    save_total_limit: int = 1
    bf16: bool = False
    fp16: bool = True
    load_best_model_at_end: Optional[bool] = None
    optim: str = "adamw_torch"
    report_to: str = "tensorboard"


def get_training_arguments(args: TrainingArgs) -> TrainingArguments:
    """Load training arguments for the model training loop.
    Args:
        args (`TrainingArgs`): The training arguments to use.
    """
    arguments = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy=args.eval_strategy,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_strategy=args.logging_strategy,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        load_best_model_at_end=args.load_best_model_at_end,
        optim=args.optim,
        report_to=args.report_to,
    )
    return arguments


def model_training(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
    eval_dataset: Optional[
        Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]
    ],
    peft_config: LoraConfig,
    training_arguments: TrainingArguments,
    dataset_text_field: str = "text",
    max_seq_length: int = 512,
) -> Dict:
    """Key function to train a model using LoRA-based fine-tuning.
    Args:
        model (`AutoModelForCausalLM`): The model to train.
        tokenizer (`AutoTokenizer`): The tokenizer to use.
        train_dataset (`torch.utils.data.Dataset`): The training dataset.
        eval_dataset (`torch.utils.data.Dataset`): The evaluation dataset.
        peft_config (`LoraConfig`): The LoRA configuration to use.
        training_arguments (`TrainingArguments`): The training arguments to use.
        dataset_text_field (`str`): The text field column in the dataset.
        max_seq_length (`int`): The maximum sequence length for the input of model.
    """
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        args=training_arguments,
        dataset_text_field=dataset_text_field,
        max_seq_length=max_seq_length,
    )

    trainer.train()
    log_history = trainer.state.log_history
    return log_history
