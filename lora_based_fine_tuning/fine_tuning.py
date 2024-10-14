from dataclasses import dataclass
from typing import Optional, Union, Dict
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer


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
