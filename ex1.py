# ex1.py
import argparse
import logging
import os
import sys
from datetime import datetime
import math

import numpy as np
import torch
import wandb
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)

# --- Metrics Calculation ---
def compute_metrics_classification(p: EvalPrediction):
    """Computes accuracy score for model predictions."""
    preds_logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds_labels = np.argmax(preds_logits, axis=1)
    true_labels = p.label_ids
    acc = accuracy_score(true_labels, preds_labels)
    return {"accuracy": acc}

# --- Main Script Logic ---
def main():
    parser = argparse.ArgumentParser()

    # --- Required arguments ---
    parser.add_argument(
        "--do_train",
        action="store_true", 
        help="Whether to run training."
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Whether to run prediction on the test set.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--lr",
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for AdamW.",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/TPU core/CPU for training and evaluation.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=-1,
        help="For debugging purposes, truncate the number of training examples.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=-1,
        help="For debugging purposes, truncate the number of evaluation examples.",
    )
    parser.add_argument(
        "--max_predict_samples",
        type=int,
        default=-1,
        help="For debugging purposes, truncate the number of prediction examples.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the trained model directory for prediction.",
    )
    
    # --- Other arguments ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="advanced-nlp-ex1-mrpc",
        help="Weights & Biases project name.",
    )


    args = parser.parse_args()
    set_seed(0)

    # --- Basic Validation ---
    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of --do_train or --do_predict must be set.")

    if args.do_predict and not args.model_path:
        raise ValueError("--model_path must be specified when using --do_predict.")

    if args.do_train and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        
    if args.do_predict and not os.path.exists(args.model_path):
         raise ValueError(f"Model path {args.model_path} does not exist.")

    # --- Model and Tokenizer ---
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # --- Load MRPC dataset ---
    logger.info("Loading MRPC dataset...")
    raw_datasets = load_dataset("glue", "mrpc")
    num_labels = raw_datasets["train"].features["label"].num_classes # Should be 2

    # --- Preprocessing ---
    def preprocess_function(examples):
        # Tokenize the sentence pairs
        # Truncation is essential, padding will be handled dynamically by DataCollator
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True, # Truncate to model's max input length
            # No padding here, let DataCollator handle it
        )

    logger.info("Preprocessing datasets...")
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=["sentence1", "sentence2", "idx"], # Remove original text columns
        desc="Running tokenizer on dataset",
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels") # Rename label column
    tokenized_datasets.set_format("torch")

    # Select subsets if needed
    if args.max_train_samples != -1 and args.do_train:
        logger.info(f"Selecting first {args.max_train_samples} training samples.")
        train_dataset = tokenized_datasets["train"].select(range(args.max_train_samples))
    else:
        train_dataset = tokenized_datasets["train"]

    if args.max_eval_samples != -1 and args.do_train:
         logger.info(f"Selecting first {args.max_eval_samples} validation samples.")
         eval_dataset = tokenized_datasets["validation"].select(range(args.max_eval_samples))
    else:
         eval_dataset = tokenized_datasets["validation"]

    if args.max_predict_samples != -1 and args.do_predict:
         logger.info(f"Selecting first {args.max_predict_samples} test samples.")
         predict_dataset = tokenized_datasets["test"].select(range(args.max_predict_samples))
    else:
        predict_dataset = tokenized_datasets["test"]


    # Data collator will dynamically pad the inputs received.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- Training ---
    if args.do_train:
        logger.info("*** Starting Training ***")

        # Initialize W&B
        run_name = f"mrpc_ep{args.num_train_epochs}_lr{args.lr}_bs{args.batch_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "learning_rate": args.lr,
                "epochs": args.num_train_epochs,
                "batch_size": args.batch_size,
                "model_name": model_name,
                # "seed": args.seed,
                "max_train_samples": args.max_train_samples,
                "max_eval_samples": args.max_eval_samples,
            }
        )

        # Load model configuration and model
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config
        )

        # --- Calculate steps per epoch ---
        num_update_steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1 # Avoid division by zero if dataset is tiny/empty
        logger.info(f"Calculated steps per epoch: {num_update_steps_per_epoch}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, run_name), # Unique output dir per run
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            # --- Evaluation and Saving Strategy ---
            eval_steps=num_update_steps_per_epoch, # Evaluate every epoch
            save_steps=num_update_steps_per_epoch, # Save checkpoint every epoch
            load_best_model_at_end=True,        # Load the best model found during training
            metric_for_best_model="accuracy",   # Metric to determine the best model
            greater_is_better=True,             # Accuracy should be maximized
            save_total_limit=1,                 # Only keep the best checkpoint
            # --- Logging ---
            # logging_strategy="steps",
            logging_steps=50,            # Log metrics every 50 steps
            report_to="wandb",           # Report metrics to W&B
            # --- Other ---
            # seed=args.seed,
            fp16=torch.cuda.is_available(), # Use mixed precision if GPU available
            push_to_hub=False,           # Do not push to Hugging Face Hub
            disable_tqdm=False,          # Show progress bars
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_classification,
        )

        # Train the model
        logger.info(f"Training with args: {training_args}")
        train_result = trainer.train()
        logger.info("Training finished.")

        # Evaluate the best model on the validation set
        logger.info("*** Evaluating Best Model on Validation Set ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        
        # Log final metrics to W&B and console
        final_val_accuracy = metrics["eval_accuracy"]
        logger.info(f"Final Validation Accuracy: {final_val_accuracy:.4f}")
        wandb.log({"final_validation_accuracy": final_val_accuracy})
        
        # Save the best model (Trainer does this automatically if load_best_model_at_end=True)
        # The best model is saved in trainer.state.best_model_checkpoint
        best_model_path = trainer.state.best_model_checkpoint
        if best_model_path:
             logger.info(f"Best model saved at {best_model_path}")
             # Optionally save it to a fixed location if needed, but Trainer already saved it.
             # trainer.save_model(os.path.join(args.output_dir, "best_model")) 
        else:
             # If load_best_model_at_end=False, or if no eval happened, save final model
             logger.info("Saving final model state.")
             final_model_path = os.path.join(training_args.output_dir, "final_model")
             trainer.save_model(final_model_path)
             logger.info(f"Final model saved at {final_model_path}")


        # Write results to res.txt
        try:
            with open("res.txt", "a") as f:
                f.write(
                    f"Epochs: {args.num_train_epochs}, LR: {args.lr}, Batch Size: {args.batch_size}, "
                    f"Val Accuracy: {final_val_accuracy:.4f}, "
                    f"Model Path: {best_model_path if best_model_path else 'N/A'}\n"
                )
            logger.info("Results appended to res.txt")
        except Exception as e:
            logger.error(f"Failed to write to res.txt: {e}")

        wandb.finish()
        logger.info("W&B Run Finished.")


    # --- Prediction ---
    if args.do_predict:
        logger.info("*** Starting Prediction ***")
        logger.info(f"Loading model from: {args.model_path}")

        # Load the specified model for prediction
        # No need to load config separately, from_pretrained handles it
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        
        # Use a dummy TrainingArguments if needed by Trainer, or configure prediction separately
        # Prediction doesn't require most training args
        predict_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, "predict_temp"), # Temporary dir for prediction outputs
            per_device_eval_batch_size=args.batch_size,
            report_to="none", # No reporting needed for prediction
            fp16=torch.cuda.is_available(),
            disable_tqdm=False,
        )

        # Initialize Trainer just for prediction
        trainer = Trainer(
            model=model,
            args=predict_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_classification # Optional for prediction, but doesn't hurt
        )

        # Run prediction
        # Note: The test set in GLUE MRPC doesn't have labels, so evaluate() won't work directly.
        # We need to use predict().
        logger.info("Running prediction on the test set...")
        predictions = trainer.predict(predict_dataset)

        # Process predictions
        preds = np.argmax(predictions.predictions, axis=1)

        # Save predictions to predictions.txt
        output_predict_file = os.path.join(args.output_dir, "predictions.txt")
        logger.info(f"Saving predictions to {output_predict_file}")
        with open(output_predict_file, "w") as writer:
            writer.write("\n".join(map(str, preds)))
            writer.write("\n") # Add trailing newline if needed

        logger.info("Prediction finished.")

if __name__ == "__main__":
    main()