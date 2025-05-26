# Comment Classification Model Finetuning

This repository contains code to finetune language models (LLaMA-2 or Qwen) for classifying comments about homelessness. The model is trained to predict multiple labels for each comment, including:

- Comment Type (Direct/Reporting)
- Critique Categories (Money Aid Allocation, Government Critique, Societal Critique)
- Response Categories (Solutions/Interventions)
- Perception Types (Personal Interaction, Media Portrayal, Not in my Backyard, Harmful Generalization, Deserving/Undeserving)
- Racist Flag

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have access to the model weights:
- For LLaMA-2: You need to request access from Meta
- For Qwen: You need to request access from Alibaba

3. Place your data files in the correct locations:
- `annotation/raw_scores.csv`: Raw annotation scores from two annotators
- `Output/annotation/soft_labels.csv`: Averaged soft labels from the annotations

## Usage

1. Run the finetuning script:
```bash
python finetune.py
```

The script will:
- Load and preprocess the data
- Split it into training and validation sets
- Finetune the model
- Save the finetuned model to `./finetuned_model`
- Print validation results

2. To use the finetuned model for predictions:
```python
from finetune import predict_comment

comment = "Your comment here"
predictions = predict_comment(comment)
```

## Model Architecture

The script uses a multi-label classification approach where each comment can have multiple labels. The model outputs probabilities for each label, which are then thresholded to get binary predictions.

## Evaluation

The model is evaluated using F1 score for each label category. The final score is the mean F1 score across all categories.

## Notes

- The script is currently configured for LLaMA-2 by default. To use Qwen, change the `model_name` variable in `finetune.py`.
- You may need to adjust the batch size and learning rate based on your available GPU memory.
- The model uses a maximum sequence length of 512 tokens. Adjust this if needed for your use case.