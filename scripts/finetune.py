import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, cohen_kappa_score
import os
from utils import (
    get_model_config,
    create_classification_prompt,
    COMMENT_TYPES,
    CRITIQUE_CATEGORIES,
    RESPONSE_CATEGORIES,
    PERCEPTION_TYPES
)

def load_and_preprocess_data():
    """Load and preprocess the raw scores and soft labels."""
    # Load raw scores
    raw_scores = pd.read_csv("annotation/raw_scores.csv")
    
    # Load soft labels
    soft_labels = pd.read_csv("Output/annotation/soft_labels.csv")
    
    # Combine the data
    combined_data = pd.merge(
        raw_scores,
        soft_labels,
        on="comment_id",
        suffixes=("_raw", "_soft")
    )
    
    return combined_data

def prepare_training_data(data):
    """Prepare the data for training by creating prompts and labels."""
    training_data = []
    
    for _, row in data.iterrows():
        # Create the classification prompt
        prompt = create_classification_prompt(row["comment"])
        
        # Create binary labels for each category
        labels = {
            "comment_type": {
                "direct": 1 if row["comment_type"] == "direct" else 0,
                "reporting": 1 if row["comment_type"] == "reporting" else 0
            },
            "critique_categories": {
                cat: row[f"{cat}_soft"] for cat in CRITIQUE_CATEGORIES
            },
            "response_categories": {
                cat: row[f"{cat}_soft"] for cat in RESPONSE_CATEGORIES
            },
            "perception_types": {
                cat: row[f"{cat}_soft"] for cat in PERCEPTION_TYPES
            },
            "racist": row["racist_soft"]
        }
        
        training_data.append({
            "prompt": prompt,
            "labels": labels
        })
    
    return training_data

def create_dataset(training_data, tokenizer):
    """Create a HuggingFace dataset from the training data."""
    def tokenize_function(examples):
        return tokenizer(
            examples["prompt"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    dataset = Dataset.from_list(training_data)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def compute_metrics(eval_pred):
    """Compute F1 score for each category."""
    predictions, labels = eval_pred
    
    # Convert predictions to binary using 0.5 threshold
    binary_preds = (predictions > 0.5).astype(int)
    
    # Calculate F1 score for each category
    f1_scores = {}
    for i, category in enumerate(COMMENT_TYPES + CRITIQUE_CATEGORIES + 
                               RESPONSE_CATEGORIES + PERCEPTION_TYPES + ["racist"]):
        f1_scores[f"f1_{category}"] = f1_score(
            labels[:, i],
            binary_preds[:, i],
            average="binary"
        )
    
    # Calculate mean F1 score
    f1_scores["mean_f1"] = np.mean(list(f1_scores.values()))
    
    return f1_scores

def calculate_agreement_metrics(predictions, human_labels):
    """Calculate agreement metrics between predictions and human labels."""
    metrics = {}
    
    # Calculate agreement for each category
    for category in (COMMENT_TYPES + CRITIQUE_CATEGORIES + 
                    RESPONSE_CATEGORIES + PERCEPTION_TYPES + ["racist"]):
        # Get binary predictions and labels for this category
        pred_binary = (predictions[category] > 0.5).astype(int)
        human_binary = human_labels[category].astype(int)
        
        # Calculate Cohen's Kappa
        kappa = cohen_kappa_score(human_binary, pred_binary)
        metrics[f"kappa_{category}"] = kappa
        
        # Calculate F1 score
        f1 = f1_score(human_binary, pred_binary, average="binary")
        metrics[f"f1_{category}"] = f1
    
    # Calculate mean metrics
    metrics["mean_kappa"] = np.mean([v for k, v in metrics.items() if k.startswith("kappa_")])
    metrics["mean_f1"] = np.mean([v for k, v in metrics.items() if k.startswith("f1_")])
    
    return metrics

def evaluate_baseline_model(model, tokenizer, val_data):
    """Evaluate the baseline (zero-shot) model performance."""
    predictions = {}
    human_labels = {}
    
    # Initialize prediction containers
    for category in (COMMENT_TYPES + CRITIQUE_CATEGORIES + 
                    RESPONSE_CATEGORIES + PERCEPTION_TYPES + ["racist"]):
        predictions[category] = []
        human_labels[category] = []
    
    # Generate predictions for each validation example
    for item in val_data:
        prompt = item["prompt"]
        labels = item["labels"]
        
        # Generate prediction
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.1
        )
        prediction_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract predictions from text (similar to utils.py extract_field)
        for category in COMMENT_TYPES:
            pred = 1 if category.lower() in prediction_text.lower() else 0
            predictions[category].append(pred)
            human_labels[category].append(labels["comment_type"][category])
        
        for category in CRITIQUE_CATEGORIES:
            pred = 1 if category.lower() in prediction_text.lower() else 0
            predictions[category].append(pred)
            human_labels[category].append(labels["critique_categories"][category])
        
        for category in RESPONSE_CATEGORIES:
            pred = 1 if category.lower() in prediction_text.lower() else 0
            predictions[category].append(pred)
            human_labels[category].append(labels["response_categories"][category])
        
        for category in PERCEPTION_TYPES:
            pred = 1 if category.lower() in prediction_text.lower() else 0
            predictions[category].append(pred)
            human_labels[category].append(labels["perception_types"][category])
        
        # Handle racist flag
        racist_pred = 1 if "racist: yes" in prediction_text.lower() else 0
        predictions["racist"].append(racist_pred)
        human_labels["racist"].append(labels["racist"])
    
    # Convert to numpy arrays
    for category in predictions:
        predictions[category] = np.array(predictions[category])
        human_labels[category] = np.array(human_labels[category])
    
    return calculate_agreement_metrics(predictions, human_labels)

def evaluate_finetuned_model(model, tokenizer, val_data):
    """Evaluate the finetuned model performance."""
    predictions = {}
    human_labels = {}
    
    # Initialize prediction containers
    for category in (COMMENT_TYPES + CRITIQUE_CATEGORIES + 
                    RESPONSE_CATEGORIES + PERCEPTION_TYPES + ["racist"]):
        predictions[category] = []
        human_labels[category] = []
    
    # Generate predictions for each validation example
    for item in val_data:
        prompt = item["prompt"]
        labels = item["labels"]
        
        # Generate prediction
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Extract predictions for each category
        for category in COMMENT_TYPES:
            pred = probs[0, COMMENT_TYPES.index(category)].item()
            predictions[category].append(pred)
            human_labels[category].append(labels["comment_type"][category])
        
        for category in CRITIQUE_CATEGORIES:
            pred = probs[0, len(COMMENT_TYPES) + CRITIQUE_CATEGORIES.index(category)].item()
            predictions[category].append(pred)
            human_labels[category].append(labels["critique_categories"][category])
        
        for category in RESPONSE_CATEGORIES:
            pred = probs[0, len(COMMENT_TYPES) + len(CRITIQUE_CATEGORIES) + 
                        RESPONSE_CATEGORIES.index(category)].item()
            predictions[category].append(pred)
            human_labels[category].append(labels["response_categories"][category])
        
        for category in PERCEPTION_TYPES:
            pred = probs[0, len(COMMENT_TYPES) + len(CRITIQUE_CATEGORIES) + 
                        len(RESPONSE_CATEGORIES) + PERCEPTION_TYPES.index(category)].item()
            predictions[category].append(pred)
            human_labels[category].append(labels["perception_types"][category])
        
        # Handle racist flag
        racist_pred = probs[0, -1].item()
        predictions["racist"].append(racist_pred)
        human_labels["racist"].append(labels["racist"])
    
    # Convert to numpy arrays
    for category in predictions:
        predictions[category] = np.array(predictions[category])
        human_labels[category] = np.array(human_labels[category])
    
    return calculate_agreement_metrics(predictions, human_labels)

def save_metrics_to_file(model_type, baseline_metrics, finetuned_metrics, output_file="model_metrics.csv"):
    """Save metrics to a CSV file with model type information."""
    # Create metrics DataFrame
    metrics_data = []
    
    # Add baseline metrics
    for metric, value in baseline_metrics.items():
        metrics_data.append({
            "model_type": model_type,
            "stage": "baseline",
            "metric": metric,
            "value": value
        })
    
    # Add finetuned metrics
    for metric, value in finetuned_metrics.items():
        metrics_data.append({
            "model_type": model_type,
            "stage": "finetuned",
            "metric": metric,
            "value": value
        })
    
    # Add improvement metrics
    for metric in baseline_metrics:
        if metric.startswith(("kappa_", "f1_")):
            improvement = finetuned_metrics[metric] - baseline_metrics[metric]
            metrics_data.append({
                "model_type": model_type,
                "stage": "improvement",
                "metric": metric,
                "value": improvement
            })
    
    # Convert to DataFrame and save
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(output_file, index=False)
    print(f"\nMetrics saved to {output_file}")

def main():
    # Load model configuration
    model_type = "llama"  # or "qwen"
    model_config = get_model_config(model_type)
    model_id = model_config["model_id"]
    
    print(f"\nTraining with {model_type.upper()} model ({model_id})")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.config.pad_token_id = model.config.eos_token_id
    
    # Load and preprocess data
    data = load_and_preprocess_data()
    training_data = prepare_training_data(data)
    
    # Split into train and validation sets
    train_data, val_data = train_test_split(
        training_data,
        test_size=0.2,
        random_state=42
    )
    
    # Evaluate baseline model
    print(f"\nEvaluating {model_type.upper()} baseline model...")
    baseline_metrics = evaluate_baseline_model(model, tokenizer, val_data)
    print(f"\n{model_type.upper()} Baseline metrics:")
    for metric, value in baseline_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create datasets
    train_dataset = create_dataset(train_data, tokenizer)
    val_dataset = create_dataset(val_data, tokenizer)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f"./finetuned_model_{model_type}",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"./logs_{model_type}",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=2e-5,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="mean_f1"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    )
    
    # Train the model
    trainer.train()
    
    # Save the model and tokenizer
    trainer.save_model(f"./finetuned_model_{model_type}")
    tokenizer.save_pretrained(f"./finetuned_model_{model_type}")
    
    # Evaluate finetuned model
    print(f"\nEvaluating {model_type.upper()} finetuned model...")
    finetuned_metrics = evaluate_finetuned_model(model, tokenizer, val_data)
    print(f"\n{model_type.upper()} Finetuned metrics:")
    for metric, value in finetuned_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Print improvement summary
    print(f"\n{model_type.upper()} Improvement summary:")
    for metric in baseline_metrics:
        if metric.startswith(("kappa_", "f1_")):
            improvement = finetuned_metrics[metric] - baseline_metrics[metric]
            print(f"{metric}: {improvement:+.4f}")
    
    # Save metrics to file
    save_metrics_to_file(model_type, baseline_metrics, finetuned_metrics)

if __name__ == "__main__":
    main() 