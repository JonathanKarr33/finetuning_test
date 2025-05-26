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
from tqdm import tqdm
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
    # Load raw scores, skipping the first row
    raw_scores = pd.read_csv("annotation/raw_scores.csv", skiprows=1)
    
    # Load soft labels
    soft_labels = pd.read_csv("output/annotation/soft_labels.csv")
    
    # Combine the data - raw_scores has the comments and city, soft_labels has the normalized scores
    combined_data = pd.concat([
        raw_scores[['City', 'Deidentified_Comment']],
        soft_labels
    ], axis=1)
    
    # Rename columns for consistency
    combined_data = combined_data.rename(columns={
        'Deidentified_Comment': 'comment',
        'City': 'city'
    })
    
    return combined_data

def prepare_training_data(data, model_type):
    """Prepare training data with model-specific formatting."""
    training_data = []
    
    for _, row in tqdm(data.iterrows(), desc="Preparing training data", total=len(data)):
        # Create the prompt
        prompt = f"Comment: {row['comment']}\nCity: {row['city']}"
        
        # Create the expected output format
        expected_output = {
            "Comment Type": row['comment_type'],
            "Critique Category": row['critique_category'],
            "Response Category": row['response_category'],
            "Perception Type": row['perception_type'],
            "Racist": "Yes" if row['racist'] > 0.5 else "No"
        }
        
        # Convert to string format
        expected_output_str = "\n".join([f"{k}: {v}" for k, v in expected_output.items()])
        
        # Format according to model type
        formatted_prompt = format_prompt_for_model(prompt, expected_output_str, model_type)
        
        training_data.append({
            "prompt": formatted_prompt,
            "expected_output": expected_output_str
        })
    
    return training_data

def create_dataset(training_data, tokenizer):
    """Create a dataset for training."""
    def tokenize_function(examples):
        # Tokenize the entire prompt including the expected output
        return tokenizer(
            examples["prompt"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
    
    # Convert to dataset format
    dataset = Dataset.from_dict({
        "prompt": [item["prompt"] for item in training_data],
        "expected_output": [item["expected_output"] for item in training_data]
    })
    
    # Tokenize
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

def print_status(message, level=0):
    """Print a status message with proper indentation."""
    indent = "  " * level
    print(f"{indent}• {message}")

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
    print_status("Generating baseline predictions...", level=1)
    for item in tqdm(val_data, desc="Baseline Evaluation"):
        prompt = item["prompt"]
        labels = item["labels"]
        
        # Generate prediction
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
        prediction_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract predictions from text
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
    print_status("Generating finetuned predictions...", level=1)
    for item in tqdm(val_data, desc="Finetuned Evaluation"):
        prompt = item["prompt"]
        labels = item["labels"]
        
        # Generate prediction
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Move inputs to GPU
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

def prepare_model_for_training(model, tokenizer, model_type):
    """Prepare the model for instruction tuning."""
    # Add special tokens if they don't exist
    special_tokens = {
        "pad_token": "<pad>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>"
    }
    
    # Add instruction tokens
    if model_type == "llama":
        special_tokens.update({
            "additional_special_tokens": [
                "### Instruction:",
                "### Input:",
                "### Response:",
                "### End"
            ]
        })
    elif model_type == "qwen":
        special_tokens.update({
            "additional_special_tokens": [
                "<|im_start|>",
                "<|im_end|>",
                "<|im_start|>system",
                "<|im_start|>user",
                "<|im_start|>assistant"
            ]
        })
    
    # Add tokens to tokenizer
    tokenizer.add_special_tokens(special_tokens)
    
    # Resize model embeddings to match new tokenizer
    model.resize_token_embeddings(len(tokenizer))
    
    # Set model to training mode
    model.train()
    
    return model, tokenizer

def format_prompt_for_model(prompt, expected_output, model_type):
    """Format the prompt according to the model's instruction format."""
    if model_type == "llama":
        return f"""### Instruction: Analyze the following comment about homelessness and provide a detailed classification.

### Input:
{prompt}

### Response:
{expected_output}

### End"""
    elif model_type == "qwen":
        return f"""<|im_start|>system
You are an expert in social behavior analysis. Your task is to analyze comments about homelessness and provide detailed classifications.
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
{expected_output}
<|im_end|>"""
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    print("\n=== Starting Finetuning Process ===")
    
    # Check GPU availability
    print_status("Checking GPU availability...")
    if torch.cuda.is_available():
        print_status(f"Using GPU: {torch.cuda.get_device_name(0)}", level=1)
        print_status(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", level=1)
    else:
        print_status("No GPU available, using CPU", level=1)
    
    # Load model configuration
    model_type = "llama"  # or "qwen"
    print_status(f"Loading {model_type.upper()} model configuration...")
    model_config = get_model_config(model_type)
    model_id = model_config["model_id"]
    print_status(f"Model ID: {model_id}", level=1)
    
    # Load tokenizer and model
    print_status("Loading tokenizer and model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        print_status("Tokenizer loaded successfully", level=1)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True  # Required for Qwen
        )
        model.config.pad_token_id = model.config.eos_token_id
        print_status(f"Model loaded on device: {model.device}", level=1)
        
        # Prepare model for instruction tuning
        model, tokenizer = prepare_model_for_training(model, tokenizer, model_type)
        print_status("Model prepared for instruction tuning", level=1)
    except Exception as e:
        print_status(f"Error loading model: {str(e)}", level=1)
        raise
    
    # Load and preprocess data
    print_status("Loading and preprocessing data...")
    try:
        data = load_and_preprocess_data()
        print_status(f"Loaded {len(data)} examples", level=1)
        
        training_data = prepare_training_data(data, model_type)
        print_status(f"Prepared {len(training_data)} training examples", level=1)
    except Exception as e:
        print_status(f"Error processing data: {str(e)}", level=1)
        raise
    
    # Split into train and validation sets
    print_status("Splitting data into train and validation sets...")
    train_data, val_data = train_test_split(
        training_data,
        test_size=0.2,
        random_state=42
    )
    print_status(f"Training set: {len(train_data)} examples", level=1)
    print_status(f"Validation set: {len(val_data)} examples", level=1)
    
    # Evaluate baseline model
    print_status(f"Evaluating {model_type.upper()} baseline model...")
    try:
        baseline_metrics = evaluate_baseline_model(model, tokenizer, val_data)
        print_status("Baseline metrics:", level=1)
        for metric, value in baseline_metrics.items():
            print_status(f"{metric}: {value:.4f}", level=2)
    except Exception as e:
        print_status(f"Error in baseline evaluation: {str(e)}", level=1)
        raise
    
    # Create datasets
    print_status("Creating training and validation datasets...")
    try:
        train_dataset = create_dataset(train_data, tokenizer)
        val_dataset = create_dataset(val_data, tokenizer)
        print_status("Datasets created successfully", level=1)
    except Exception as e:
        print_status(f"Error creating datasets: {str(e)}", level=1)
        raise
    
    # Set up training arguments
    print_status("Setting up training configuration...")
    training_args = TrainingArguments(
        output_dir=f"finetuned_{model_type}",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=True,  # Enable mixed precision training
        gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
        gradient_checkpointing=True,  # Save memory by checkpointing gradients
        learning_rate=2e-5,
        max_grad_norm=1.0,  # Gradient clipping
        # Generation-specific arguments
        generation_max_length=512,
        generation_num_beams=4,
        predict_with_generate=True,
        # Instruction tuning specific
        remove_unused_columns=False,  # Keep all columns for generation
        label_names=["input_ids"],  # Use input_ids as labels for causal LM
    )
    
    # Initialize trainer
    print_status("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Use causal language modeling
        )
    )
    
    # Train the model
    print_status("Starting training...")
    trainer.train()
    
    # Save the model and tokenizer
    print_status("Saving model and tokenizer...")
    trainer.save_model(f"finetuned_{model_type}")
    tokenizer.save_pretrained(f"finetuned_{model_type}")
    
    # Evaluate the model
    print_status("Evaluating model...")
    metrics = trainer.evaluate()
    print_status(f"Final metrics: {metrics}", level=1)
    
    # Save metrics
    save_metrics_to_file(model_type, baseline_metrics, metrics)
    
    print_status("Finetuning complete!")

if __name__ == "__main__":
    main() 