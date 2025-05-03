from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments

@dataclass
class DataTrainingArguments:
    """
    Arguments related to what data we input for training and eval.
    """
    max_train_samples: int = field(
        default=-1,
        metadata={"help": "Number of samples to use for training, or -1 to use all."}
    )
    max_eval_samples: int = field(
        default=-1,
        metadata={"help": "Number of samples to use for evaluation, or -1 to use all."}
    )
    max_predict_samples: int = field(
        default=-1,
        metadata={"help": "Number of samples to use for prediction, or -1 to use all."}
    )

@dataclass
class RunArguments:
    """
    Flags and hyperparameters for running.
    """
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Total number of training epochs to perform."}
    )
    lr: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for AdamW."}
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per device for training and evaluation."}
    )
    do_train: bool = field(
        default=False,
        metadata={"help": "Whether to run training."}
    )
    do_predict: bool = field(
        default=False,
        metadata={"help": "Whether to run predictions on the test set."}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, RunArguments))
    model_args, data_args, run_args = parser.parse_args_into_dataclasses()

    training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=run_args.num_train_epochs,        # from --num_train_epochs
    learning_rate=run_args.lr,                         # from --lr
    per_device_train_batch_size=run_args.batch_size,   # from --batch_size
    per_device_eval_batch_size=run_args.batch_size,    # from --batch_size
    do_train=run_args.do_train,                        # from --do_train
    do_predict=run_args.do_predict,                    # from --do_predict
)
    # print(training_args)


if __name__ == "__main__":
    main()
