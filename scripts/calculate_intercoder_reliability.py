import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import os

# Create output directories if they don't exist
os.makedirs('output/charts', exist_ok=True)

# Mapping from soft label columns to classification columns
field_map = {
    'Direct': 'Comment_direct',
    'Reporting': 'Comment_reporting',
    'money aid allocation': 'Critique_money aid allocation',
    'government critique': 'Critique_government critique',
    'societal critique': 'Critique_societal critique',
    'solutions/interventions': 'Response_solutions/interventions',
    'personal interaction': 'Perception_personal interaction',
    'media portrayal': 'Perception_media portrayal',
    'not in my backyard': 'Perception_not in my backyard',
    'harmful generalization': 'Perception_harmful generalization',
    'deserving/undeserving': 'Perception_deserving/undeserving',
    'Racist': 'Racist'
}

def load_classifications():
    """Load both Llama and Qwen classification results for original and mitigated data."""
    try:
        print("Loading classification files...")
        llama_df = pd.read_csv("output/classified_comments_llama.csv")
        qwen_df = pd.read_csv("output/classified_comments_qwen.csv")
        llama_mit_df = pd.read_csv("output/mitigated_comments_llama.csv")
        qwen_mit_df = pd.read_csv("output/mitigated_comments_qwen.csv")
        soft_labels_df = pd.read_csv("output/annotation/soft_labels.csv")
        print(f"Loaded {len(llama_df)} comments from each model")
        print(f"Mitigated data lengths - Llama: {len(llama_mit_df)}, Qwen: {len(qwen_mit_df)}")
        print(f"Soft labels length: {len(soft_labels_df)}")
        print("\nSoft labels columns:")
        print(soft_labels_df.columns.tolist())
        print("\nLlama classification columns:")
        print(llama_df.columns.tolist())
        print("\nQwen classification columns:")
        print(qwen_df.columns.tolist())
        return llama_df, qwen_df, llama_mit_df, qwen_mit_df, soft_labels_df
    except Exception as e:
        print(f"Error loading classification files: {e}")
        exit(1)

def calculate_kappa(llama_df, qwen_df, field):
    """Calculate Cohen's Kappa for a specific field."""
    # Get the values for both models
    llama_values = llama_df[field].fillna(0)  # Fill NaN with 0 for flag fields
    qwen_values = qwen_df[field].fillna(0)
    
    # Convert to int for flag fields (except Racist)
    if field != "Racist":
        llama_values = llama_values.astype(int)
        qwen_values = qwen_values.astype(int)
    else:
        # For Racist, extract the value from the text
        def extract_racist_value(text):
            if pd.isna(text):
                return 0
            text = str(text).lower().strip()
            if text == "yes" or "racist: yes" in text:
                return 1  # Yes is positive (1)
            elif text == "no" or "racist: no" in text:
                return 0  # No is negative (0)
            return 0  # Default to negative (0) if not found
        
        llama_values = llama_values.apply(extract_racist_value)
        qwen_values = qwen_values.apply(extract_racist_value)
    
    # Calculate kappa
    kappa = cohen_kappa_score(llama_values, qwen_values)
    
    # Create confusion matrix with explicit labels
    cm = confusion_matrix(llama_values, qwen_values, labels=[0, 1])
    
    # Flip the matrix to put positive class (1) in top left
    cm = np.flip(cm, axis=(0, 1))
    
    # Calculate total positives for each model
    llama_positives = cm[0, 0] + cm[0, 1]  # Top row (1s)
    qwen_positives = cm[0, 0] + cm[1, 0]   # Left column (1s)
    
    return kappa, cm, llama_positives, qwen_positives

def calculate_soft_label_agreement(llama_df, qwen_df, soft_labels_df, field):
    """Calculate agreement between models and soft labels."""
    # Get the values for both models and soft labels, ensuring numeric dtype
    llama_values = pd.to_numeric(llama_df[field_map[field]], errors='coerce').fillna(0)
    qwen_values = pd.to_numeric(qwen_df[field_map[field]], errors='coerce').fillna(0)
    soft_values = pd.to_numeric(soft_labels_df[field], errors='coerce').fillna(0)

    # Binarize soft labels for confusion matrix
    soft_values_bin = (soft_values >= 0.5).astype(int)

    # Calculate agreement metrics
    llama_agreement = np.mean(np.abs(llama_values - soft_values) <= 0.5)  # Within 0.5
    qwen_agreement = np.mean(np.abs(qwen_values - soft_values) <= 0.5)    # Within 0.5

    # Create confusion matrices for soft labels (binarized)
    llama_cm = confusion_matrix(soft_values_bin, llama_values, labels=[0, 1])
    qwen_cm = confusion_matrix(soft_values_bin, qwen_values, labels=[0, 1])

    return {
        'llama_agreement': llama_agreement,
        'qwen_agreement': qwen_agreement,
        'llama_cm': llama_cm,
        'qwen_cm': qwen_cm,
        'soft_values': soft_values,
        'llama_values': llama_values,
        'qwen_values': qwen_values
    }

def plot_confusion_matrix(cm, field, kappa, llama_positives, qwen_positives, ax, delta_info=None):
    """Plot confusion matrix on the given axis."""
    # Get labels based on field type
    if field == "Racist":
        labels = ["Yes", "No"]  # Yes (1) is positive, No (0) is negative
    else:
        labels = ["1", "0"]  # 1 is positive, 0 is negative
    
    # Create heatmap with labels and consistent vmax
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax, cbar=True, vmax=500)
    
    # Format field name for display
    if field == "Racist":
        title = f"Racist\nCohen's κ = {kappa:.2f}"
    elif field.startswith("Comment_"):
        category = "Direct Comment" if field == "Comment_direct" else "Reporting Comment"
        title = f"Comment: {category}\nCohen's κ = {kappa:.2f}"
    elif field.startswith("Critique_"):
        category = field.replace("Critique_", "").replace("_", " ").title()
        title = f"Critique: {category}\nCohen's κ = {kappa:.2f}"
    elif field.startswith("Response_"):
        category = field.replace("Response_", "").replace("_", " ").title()
        title = f"Response: {category}\nCohen's κ = {kappa:.2f}"
    elif field.startswith("Perception_"):
        category = field.replace("Perception_", "").replace("_", " ").title()
        title = f"Perception: {category}\nCohen's κ = {kappa:.2f}"
    else:
        # Default case - use the field name directly
        title = f"{field}\nCohen's κ = {kappa:.2f}"
    
    # Add delta information if provided
    if delta_info:
        title += f"\nΔκ = {delta_info['kappa_delta']:+.2f}"
        title += f"\nΔLlama +: {delta_info['llama_delta']:+d}"
        title += f"\nΔQwen +: {delta_info['qwen_delta']:+d}"
    else:
        title += f"\nLlama +: {llama_positives}"
        title += f"\nQwen +: {qwen_positives}"
    
    # Add title with kappa score and positive counts
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Qwen Predictions', fontweight='bold')
    ax.set_ylabel('Llama Predictions', fontweight='bold')

def plot_delta_heatmap(original_results, mitigated_results, pdf_path):
    """Create a heatmap showing the changes in kappa scores."""
    fields = list(original_results.keys())
    kappa_deltas = [mitigated_results[f]['kappa'] - original_results[f]['kappa'] for f in fields]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create heatmap
    sns.heatmap(np.array(kappa_deltas).reshape(-1, 1), 
                annot=True, 
                fmt='+.2f',
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Δκ'},
                yticklabels=fields)
    
    plt.title('Changes in Cohen\'s Kappa Scores After Mitigation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save to PDF
    with PdfPages(pdf_path) as pdf:
        pdf.savefig()
        plt.close()

def plot_soft_label_comparison(soft_results, pdf_path):
    """Create visualization comparing model predictions with soft labels."""
    fields = list(soft_results.keys())
    
    # Compute agreement breakdowns for each model and field
    llama_pos, llama_neg, llama_fp, llama_fn = [], [], [], []
    qwen_pos, qwen_neg, qwen_fp, qwen_fn = [], [], [], []
    soft_05 = []  # Single list for soft label 0.5 cases
    for f in fields:
        soft = np.array(soft_results[f]['soft_values'])
        llama = np.array(soft_results[f]['llama_values'])
        qwen = np.array(soft_results[f]['qwen_values'])
        
        # Positive agreement: model==1 & soft==1
        llama_pos.append(np.mean((llama == 1) & (soft == 1)))
        qwen_pos.append(np.mean((qwen == 1) & (soft == 1)))
        
        # Negative agreement: model==0 & soft==0
        llama_neg.append(np.mean((llama == 0) & (soft == 0)))
        qwen_neg.append(np.mean((qwen == 0) & (soft == 0)))
        
        # Soft label is 0.5 (same for both models)
        soft_05.append(np.mean(soft == 0.5))
        
        # False positives: model==1 & soft==0
        llama_fp.append(np.mean((soft == 0) & (llama == 1)))
        qwen_fp.append(np.mean((soft == 0) & (qwen == 1)))
        
        # False negatives: model==0 & soft==1
        llama_fn.append(np.mean((soft == 1) & (llama == 0)))
        qwen_fn.append(np.mean((soft == 1) & (qwen == 0)))
    
    # Prepare DataFrame for grouped, stacked bar chart
    agreement_df = pd.DataFrame({
        'Field': fields,
        'Llama Positive Agreement': llama_pos,
        'Llama Negative Agreement': llama_neg,
        'Llama False Positive': llama_fp,
        'Llama False Negative': llama_fn,
        'Soft Label 0.5': soft_05,  # Single column for soft label 0.5
        'Qwen Positive Agreement': qwen_pos,
        'Qwen Negative Agreement': qwen_neg,
        'Qwen False Positive': qwen_fp,
        'Qwen False Negative': qwen_fn
    })
    
    # Plot grouped, stacked bars
    fig, ax1 = plt.subplots(figsize=(9, 9))  # Changed to square with full width
    width = 0.35
    x = np.arange(len(fields))
    
    # Llama bars (reordered stacking)
    ax1.bar(x - width/2, agreement_df['Soft Label 0.5'], width, color='yellow', label='Soft Label 0.5', bottom=0)
    ax1.bar(x - width/2, agreement_df['Llama Positive Agreement'], width, color='lightblue', label='Llama Positive Agreement', 
            bottom=agreement_df['Soft Label 0.5'])
    ax1.bar(x - width/2, agreement_df['Llama Negative Agreement'], width, color='blue', label='Llama Negative Agreement', 
            bottom=agreement_df['Soft Label 0.5'] + agreement_df['Llama Positive Agreement'])
    ax1.bar(x - width/2, agreement_df['Llama False Positive'], width, color='orange', label='Llama False Positive', 
            bottom=agreement_df['Soft Label 0.5'] + agreement_df['Llama Positive Agreement'] + agreement_df['Llama Negative Agreement'])
    ax1.bar(x - width/2, agreement_df['Llama False Negative'], width, color='darkred', label='Llama False Negative', 
            bottom=agreement_df['Soft Label 0.5'] + agreement_df['Llama Positive Agreement'] + agreement_df['Llama Negative Agreement'] + agreement_df['Llama False Positive'])
    
    # Qwen bars (reordered stacking)
    ax1.bar(x + width/2, agreement_df['Soft Label 0.5'], width, color='yellow', label='Soft Label 0.5', bottom=0)
    ax1.bar(x + width/2, agreement_df['Qwen Positive Agreement'], width, color='lightgreen', label='Qwen Positive Agreement', 
            bottom=agreement_df['Soft Label 0.5'])
    ax1.bar(x + width/2, agreement_df['Qwen Negative Agreement'], width, color='green', label='Qwen Negative Agreement', 
            bottom=agreement_df['Soft Label 0.5'] + agreement_df['Qwen Positive Agreement'])
    ax1.bar(x + width/2, agreement_df['Qwen False Positive'], width, color='gold', label='Qwen False Positive', 
            bottom=agreement_df['Soft Label 0.5'] + agreement_df['Qwen Positive Agreement'] + agreement_df['Qwen Negative Agreement'])
    ax1.bar(x + width/2, agreement_df['Qwen False Negative'], width, color='red', label='Qwen False Negative', 
            bottom=agreement_df['Soft Label 0.5'] + agreement_df['Qwen Positive Agreement'] + agreement_df['Qwen Negative Agreement'] + agreement_df['Qwen False Positive'])
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(fields, rotation=45, ha='right', fontsize=12)
    ax1.set_title('LLM Agreement with Soft Labels', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Proportion', fontsize=14)
    ax1.set_ylim(0, 1)
    
    # Custom legend (removing duplicate Soft Label 0.5 entry)
    handles = [
        plt.Rectangle((0,0),1,1,color='yellow'),
        plt.Rectangle((0,0),1,1,color='lightblue'),
        plt.Rectangle((0,0),1,1,color='blue'),
        plt.Rectangle((0,0),1,1,color='orange'),
        plt.Rectangle((0,0),1,1,color='darkred'),
        plt.Rectangle((0,0),1,1,color='lightgreen'),
        plt.Rectangle((0,0),1,1,color='green'),
        plt.Rectangle((0,0),1,1,color='gold'),
        plt.Rectangle((0,0),1,1,color='red')
    ]
    labels = [
        'Soft Label 0.5',
        'Llama Positive Agreement', 'Llama Negative Agreement', 'Llama False Positive', 'Llama False Negative',
        'Qwen Positive Agreement', 'Qwen Negative Agreement', 'Qwen False Positive', 'Qwen False Negative'
    ]
    ax1.legend(handles, labels, bbox_to_anchor=(0.5, -0.35), loc='upper center', ncol=3, fontsize=12)
    plt.tight_layout()
    
    # Save agreement chart to PDF
    with PdfPages('output/charts/agreement.pdf') as pdf:
        pdf.savefig(fig)
        plt.close()
    
    # Plot distribution of soft labels with updated labels
    soft_dist = pd.DataFrame({
        'Field': fields,
        'Negative Agreement (0)': [np.mean(soft_results[f]['soft_values'] == 0) for f in fields],
        'Soft Label 0.5': [np.mean(soft_results[f]['soft_values'] == 0.5) for f in fields],
        'Positive Agreement (1)': [np.mean(soft_results[f]['soft_values'] == 1) for f in fields]
    })
    fig, ax2 = plt.subplots(figsize=(9, 7))
    soft_dist.plot(x='Field', y=['Negative Agreement (0)', 'Soft Label 0.5', 'Positive Agreement (1)'], 
                  kind='bar', stacked=True, ax=ax2, rot=45, color=['green', 'yellow', 'red'])
    ax2.set_title('Agreement Between Annotators', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Proportion', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    
    # Save soft label distribution chart to PDF
    with PdfPages('output/charts/distribution.pdf') as pdf:
        pdf.savefig(fig)
        plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate inter-coder reliability between Llama and Qwen models')
    parser.add_argument('--include_mitigation', action='store_true', help='Include mitigation analysis')
    args = parser.parse_args()
    
    # Load the classification results
    llama_df, qwen_df, llama_mit_df, qwen_mit_df, soft_labels_df = load_classifications()
    
    # Ensure we're comparing the same comments
    assert len(llama_df) == len(qwen_df), "Number of comments must match between Llama and Qwen"
    
    # Fields to analyze - these match the column names in soft_labels.csv
    fields = list(field_map.keys())
    
    # Calculate agreement statistics for original data
    print("\nCalculating agreement statistics for original data...")
    original_results = {}
    for field in tqdm(fields, desc="Fields"):
        kappa, cm, llama_positives, qwen_positives = calculate_kappa(
            llama_df, qwen_df, field_map[field]
        )
        original_results[field] = {
            'kappa': kappa,
            'confusion_matrix': cm,
            'llama_positives': llama_positives,
            'qwen_positives': qwen_positives
        }
    
    # Calculate agreement with soft labels
    print("\nCalculating agreement with soft labels...")
    soft_results = {}
    for field in tqdm(fields, desc="Fields"):
        soft_results[field] = calculate_soft_label_agreement(
            llama_df, qwen_df, soft_labels_df, field
        )
    
    # Create PDF for original data
    with PdfPages('output/charts/confusion_matrices.pdf') as pdf:
        # Calculate grid dimensions
        n_fields = len(original_results)
        n_cols = 3
        n_rows = (n_fields + n_cols - 1) // n_cols
        
        # Create figure with subplots - add extra height for title and row spacing
        fig = plt.figure(figsize=(15, 5 * n_rows + 1))
        
        # Add main title
        plt.suptitle('LLM Classification of Original Data', fontsize=16, fontweight='bold', y=0.98)
        
        # Create a gridspec with extra space at top and between rows
        gs = plt.GridSpec(n_rows, n_cols, top=0.9, hspace=0.6)  # Increased hspace from 0.4 to 0.6
        
        # Plot each confusion matrix
        for idx, (field, stats) in enumerate(original_results.items(), 1):
            row = (idx - 1) // n_cols
            col = (idx - 1) % n_cols
            ax = plt.subplot(gs[row, col])
            
            plot_confusion_matrix(
                stats['confusion_matrix'], 
                field, 
                stats['kappa'],
                stats['llama_positives'],
                stats['qwen_positives'],
                ax
            )
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    # Create soft label comparison visualization
    plot_soft_label_comparison(soft_results, 'output/charts/soft_label_comparison.pdf')
    
    # Calculate agreement statistics for mitigated data
    print("\nCalculating agreement statistics for mitigated data...")
    mitigated_results = {}
    for field in tqdm(fields, desc="Fields"):
        kappa, cm, llama_positives, qwen_positives = calculate_kappa(
            llama_mit_df, qwen_mit_df, field_map[field]
        )
        mitigated_results[field] = {
            'kappa': kappa,
            'confusion_matrix': cm,
            'llama_positives': llama_positives,
            'qwen_positives': qwen_positives
        }
    
    # Create PDF for mitigated data
    with PdfPages('output/charts/mitigated_confusion_matrices.pdf') as pdf:
        # Calculate grid dimensions
        n_fields = len(mitigated_results)
        n_cols = 3
        n_rows = (n_fields + n_cols - 1) // n_cols
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 5 * n_rows + 1))
        
        # Add main title
        plt.suptitle('LLM Classification of Mitigated Data', fontsize=16, fontweight='bold', y=0.98)
        
        # Create a gridspec with extra space at top and between rows
        gs = plt.GridSpec(n_rows, n_cols, top=0.9, hspace=0.6)  # Increased hspace from 0.4 to 0.6
        
        # Plot each confusion matrix
        for idx, (field, stats) in enumerate(mitigated_results.items(), 1):
            row = (idx - 1) // n_cols
            col = (idx - 1) % n_cols
            ax = plt.subplot(gs[row, col])
            
            # Calculate deltas for display
            delta_info = {
                'kappa_delta': stats['kappa'] - original_results[field]['kappa'],
                'llama_delta': stats['llama_positives'] - original_results[field]['llama_positives'],
                'qwen_delta': stats['qwen_positives'] - original_results[field]['qwen_positives']
            }
            
            plot_confusion_matrix(
                stats['confusion_matrix'], 
                field, 
                stats['kappa'],
                stats['llama_positives'],
                stats['qwen_positives'],
                ax,
                delta_info
            )
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    # Create delta heatmap
    plot_delta_heatmap(original_results, mitigated_results, 'output/charts/kappa_deltas.pdf')
    
    # Print results
    print("\nInter-coder Reliability Results:")
    print("=" * 50)
    for field in fields:
        print(f"\n{field}:")
        print(f"Original Kappa: {original_results[field]['kappa']:.3f}")
        print(f"Agreement with Soft Labels - Llama: {soft_results[field]['llama_agreement']:.3f}, Qwen: {soft_results[field]['qwen_agreement']:.3f}")
        print(f"Mitigated Kappa: {mitigated_results[field]['kappa']:.3f}")
        print(f"Δκ: {mitigated_results[field]['kappa'] - original_results[field]['kappa']:+.3f}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Field': fields,
        'Original_Kappa': [original_results[f]['kappa'] for f in fields],
        'Original_Llama_Positives': [original_results[f]['llama_positives'] for f in fields],
        'Original_Qwen_Positives': [original_results[f]['qwen_positives'] for f in fields],
        'Llama_Soft_Agreement': [soft_results[f]['llama_agreement'] for f in fields],
        'Qwen_Soft_Agreement': [soft_results[f]['qwen_agreement'] for f in fields],
        'Soft_Label_0.5': [np.mean(soft_results[f]['soft_values'] == 0.5) for f in fields],
        'Soft_Label_1': [np.mean(soft_results[f]['soft_values'] == 1) for f in fields],
        'Soft_Label_0': [np.mean(soft_results[f]['soft_values'] == 0) for f in fields],
        'Llama_Positive_Agreement': [np.mean((soft_results[f]['llama_values'] == 1) & (soft_results[f]['soft_values'] == 1)) for f in fields],
        'Llama_Negative_Agreement': [np.mean((soft_results[f]['llama_values'] == 0) & (soft_results[f]['soft_values'] == 0)) for f in fields],
        'Llama_False_Positive': [np.mean((soft_results[f]['soft_values'] == 0) & (soft_results[f]['llama_values'] == 1)) for f in fields],
        'Llama_False_Negative': [np.mean((soft_results[f]['soft_values'] == 1) & (soft_results[f]['llama_values'] == 0)) for f in fields],
        'Qwen_Positive_Agreement': [np.mean((soft_results[f]['qwen_values'] == 1) & (soft_results[f]['soft_values'] == 1)) for f in fields],
        'Qwen_Negative_Agreement': [np.mean((soft_results[f]['qwen_values'] == 0) & (soft_results[f]['soft_values'] == 0)) for f in fields],
        'Qwen_False_Positive': [np.mean((soft_results[f]['soft_values'] == 0) & (soft_results[f]['qwen_values'] == 1)) for f in fields],
        'Qwen_False_Negative': [np.mean((soft_results[f]['soft_values'] == 1) & (soft_results[f]['qwen_values'] == 0)) for f in fields],
        'Mitigated_Kappa': [mitigated_results[f]['kappa'] for f in fields],
        'Mitigated_Llama_Positives': [mitigated_results[f]['llama_positives'] for f in fields],
        'Mitigated_Qwen_Positives': [mitigated_results[f]['qwen_positives'] for f in fields],
        'Kappa_Delta': [mitigated_results[f]['kappa'] - original_results[f]['kappa'] for f in fields],
        'Llama_Positives_Delta': [mitigated_results[f]['llama_positives'] - original_results[f]['llama_positives'] for f in fields],
        'Qwen_Positives_Delta': [mitigated_results[f]['qwen_positives'] - original_results[f]['qwen_positives'] for f in fields]
    })
    results_df.to_csv("output/charts/intercoder_reliability_results.csv", index=False)
    print("\nResults saved to output/charts/intercoder_reliability_results.csv")

    # Create new CSV with counts
    counts_df = pd.DataFrame({
        'Field': fields,
        'Soft_Label_0.5_Count': [np.sum(soft_results[f]['soft_values'] == 0.5) for f in fields],
        'Soft_Label_1_Count': [np.sum(soft_results[f]['soft_values'] == 1) for f in fields],
        'Soft_Label_0_Count': [np.sum(soft_results[f]['soft_values'] == 0) for f in fields],
        'Llama_Positive_Count': [np.sum(soft_results[f]['llama_values'] == 1) for f in fields],
        'Llama_Negative_Count': [np.sum(soft_results[f]['llama_values'] == 0) for f in fields],
        'Qwen_Positive_Count': [np.sum(soft_results[f]['qwen_values'] == 1) for f in fields],
        'Qwen_Negative_Count': [np.sum(soft_results[f]['qwen_values'] == 0) for f in fields],
        'Mitigated_Llama_Positive_Count': [np.sum(llama_mit_df[field_map[f]] == 1) for f in fields],
        'Mitigated_Llama_Negative_Count': [np.sum(llama_mit_df[field_map[f]] == 0) for f in fields],
        'Mitigated_Qwen_Positive_Count': [np.sum(qwen_mit_df[field_map[f]] == 1) for f in fields],
        'Mitigated_Qwen_Negative_Count': [np.sum(qwen_mit_df[field_map[f]] == 0) for f in fields]
    })
    counts_df.to_csv("output/charts/category_counts.csv", index=False)
    print("Category counts saved to output/charts/category_counts.csv")

if __name__ == "__main__":
    main() 