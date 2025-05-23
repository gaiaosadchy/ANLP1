from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    TrainingArguments, 
    AutoConfig, AutoTokenizer, 
    AutoModelForSequenceClassification,
    EvalPrediction,
    Trainer,
    DataCollatorWithPadding,
)   
import os
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import wandb
import numpy as np


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
    wandb_project: str = field(
        default="my-default-project",
        metadata={"help": "Weights & Biases project name"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, RunArguments))
    model_args, data_args, run_args = parser.parse_args_into_dataclasses()

    run_name = f"mrpc_ep{run_args.num_train_epochs}_lr{run_args.lr}_bs{run_args.batch_size}"

    training_args = TrainingArguments(
        output_dir=os.path.join("results", run_name),
        num_train_epochs=run_args.num_train_epochs,        
        learning_rate=run_args.lr,                         
        per_device_train_batch_size=run_args.batch_size,   
        # per_device_eval_batch_size=run_args.batch_size,    
        do_train=run_args.do_train,                     
        do_predict=run_args.do_predict,           

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,           
        report_to="wandb",   
        disable_tqdm=False,  
    )

    raw_dataset = load_dataset('glue', 'mrpc')

    train_dataset = (
        raw_dataset["train"].select(range(data_args.max_train_samples))
        if data_args.max_train_samples != -1 and run_args.do_train
        else raw_dataset["train"]
    )
    eval_dataset = (
        raw_dataset["validation"].select(range(data_args.max_eval_samples))
        if data_args.max_eval_samples != -1 and run_args.do_train
        else raw_dataset["validation"]
    )
    test_dataset = (
        raw_dataset["test"].select(range(data_args.max_predict_samples))
        if data_args.max_predict_samples != -1 and run_args.do_predict
        else raw_dataset["test"]
    )
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    def preprocess_function(examples):
        result = tokenizer(examples['sentence1'], examples['sentence2'], max_length=512, truncation=True, padding=False)
        return result
    
    text_cols = ['sentence1', 'sentence2']
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=text_cols + (['idx'] if 'idx' in train_dataset.column_names else [])
    )
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=text_cols,
    )
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=text_cols
    )

    def compute_metrics_classification(p: EvalPrediction):
        """Computes accuracy score for model predictions."""
        preds_logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds_labels = np.argmax(preds_logits, axis=1)
        true_labels = p.label_ids
        acc = accuracy_score(true_labels, preds_labels)
        return {"accuracy": acc}
    

    if training_args.do_train:

        wandb.login()
        wandb.init(
            project=model_args.wandb_project,
            name=run_name,
            config={
                "learning_rate": run_args.lr,
                "epochs": run_args.num_train_epochs,
                "batch_size": run_args.batch_size,
                "model_name": 'bert-base-uncased',
                "max_train_samples": data_args.max_train_samples,
                "max_eval_samples": data_args.max_eval_samples,
            }
        )

        config = AutoConfig.from_pretrained('bert-base-uncased')
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_classification,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        train_result = trainer.train()
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        final_val_accuracy = metrics["eval_accuracy"]
        wandb.log({"final_validation_accuracy": final_val_accuracy})
        final_model_path = os.path.join(training_args.output_dir, "final_model")
        trainer.save_model(final_model_path)

        try:
            with open("res.txt", "a") as f:
                f.write(
                    f"epoch_num: {run_args.num_train_epochs}, lr: {run_args.lr}, batch_size: {run_args.batch_size}, "
                    f"eval_acc: {final_val_accuracy:.4f}\n"
                )
        except Exception as e:
            print(f"Error writing to file: {e}")

        wandb.finish()
    
    if training_args.do_predict:
        model = AutoModelForSequenceClassification.from_pretrained(model_args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)

        predict_args = TrainingArguments(
            output_dir=os.path.join(training_args.output_dir, "predict_temp"),
            # per_device_eval_batch_size=run_args.batch_size,
            report_to="none",
            disable_tqdm=False,
        )

        trainer = Trainer(
            model=model,
            args=predict_args,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_classification
        )

        test_preds = trainer.predict(test_dataset)
        test_acc = test_preds.metrics["test_accuracy"]
        wandb.init(project=model_args.wandb_project, name=run_name+"_test", resume="allow")
        wandb.log({"test_accuracy": test_acc})
        wandb.finish()
        test_labels = np.argmax(test_preds.predictions, axis=1)

        raw_test = raw_dataset["test"].select(
        range(data_args.max_predict_samples)
        ) if data_args.max_predict_samples != -1 else raw_dataset["test"]

        output_predict_file = os.path.join(training_args.output_dir, "predictions.txt")
        with open(output_predict_file, "w") as writer:
            for s1, s2, label in zip(raw_test["sentence1"],
                                  raw_test["sentence2"],
                                  test_labels):
                 writer.write(f"{s1}###{s2}###{label}\n")
        
        eval_preds = trainer.predict(eval_dataset)
        eval_acc = eval_preds.metrics["test_accuracy"]
        eval_labels = np.argmax(eval_preds.predictions, axis=1)

        output_eval_file = os.path.join(training_args.output_dir, "eval_predictions.txt")
        with open(output_eval_file, "w") as writer:
            for s1, s2, label in zip(raw_dataset["validation"]["sentence1"],
                                  raw_dataset["validation"]["sentence2"],
                                  eval_labels):
                 writer.write(f"{s1}###{s2}###{label}\n")


if __name__ == "__main__":
    main()
