"""
Bi-Encoder Baseline Evaluation (Zero-Shot vs Fine-Tuned)
Αξιολογεί την αρχιτεκτονική Bi-Encoder (π.χ. MiniLM) χρησιμοποιώντας τα 
σύνολα Train, Valid και Test, αποτρέποντας τη διαρροή δεδομένων (Data Leakage).
"""

import os
import re
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# ΡΥΘΜΙΣΕΙΣ ΠΕΙΡΑΜΑΤΟΣ
# ==========================================
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
DATASET_PATH = "./data/Walmart-Amazon" 
TRAIN_FRACTION = 0.30  # Χρησιμοποιούμε μόνο το 30% του Train set
EPOCHS = 4
BATCH_SIZE = 16

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s\.\-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def load_and_format_data(data_path, split_name, frac=1.0):
    """Φορτώνει και ενώνει τα δεδομένα. Επιστρέφει DataFrame με text1, text2, label"""
    src = pd.read_csv(os.path.join(data_path, "source.csv"))
    tgt = pd.read_csv(os.path.join(data_path, "target.csv"))
    split_df = pd.read_csv(os.path.join(data_path, f"{split_name}.csv"))
    
    # Επιλογή τυχαίου υποσυνόλου μόνο για το train set
    if split_name == "train" and frac < 1.0:
        split_df = split_df.sample(frac=frac, random_state=42)
        
    merged = split_df.merge(src, left_on="ltable_id", right_on="id") \
                     .merge(tgt, left_on="rtable_id", right_on="id", suffixes=('_l', '_r'))
    
    # Walmart-Amazon Columns: Title, ModelNo, Brand, Category, Price
    columns = ["title", "modelno", "brand", "category", "price"]
    
    texts1, texts2, labels = [], [], []
    for _, row in merged.iterrows():
        # Απλή συνένωση πεδίων
        val_l = " ".join([clean_text(row.get(f"{col}_l", "")) for col in columns])
        val_r = " ".join([clean_text(row.get(f"{col}_r", "")) for col in columns])
        
        texts1.append(val_l)
        texts2.append(val_r)
        labels.append(int(row['label']))
        
    return pd.DataFrame({"text1": texts1, "text2": texts2, "label": labels})

def evaluate_model(model, df_valid, df_test):
    """Υπολογίζει το Cosine Similarity. Βρίσκει Threshold στο Valid, μετράει στο Test."""
    # 1. Παραγωγή Embeddings
    print("   -> Encoding Validation set...")
    emb1_val = model.encode(df_valid['text1'].tolist(), convert_to_tensor=True, show_progress_bar=False)
    emb2_val = model.encode(df_valid['text2'].tolist(), convert_to_tensor=True, show_progress_bar=False)
    sim_val = torch.nn.functional.cosine_similarity(emb1_val, emb2_val).cpu().numpy()
    
    print("   -> Encoding Test set...")
    emb1_test = model.encode(df_test['text1'].tolist(), convert_to_tensor=True, show_progress_bar=False)
    emb2_test = model.encode(df_test['text2'].tolist(), convert_to_tensor=True, show_progress_bar=False)
    sim_test = torch.nn.functional.cosine_similarity(emb1_test, emb2_test).cpu().numpy()
    
    # 2. Threshold Tuning validation set (αποφυγή Data Leakage)
    best_thresh, best_val_f1 = 0.5, 0
    for t in np.arange(0.2, 0.95, 0.05):
        preds = (sim_val >= t).astype(int)
        f1 = f1_score(df_valid['label'], preds, zero_division=0)
        if f1 > best_val_f1:
            best_val_f1, best_thresh = f1, t
            
    # 3. Τελική Αξιολόγηση test set με το βέλτιστο threshold
    test_preds = (sim_test >= best_thresh).astype(int)
    
    metrics = {
        "Threshold": best_thresh,
        "Precision": precision_score(df_test['label'], test_preds, zero_division=0),
        "Recall": recall_score(df_test['label'], test_preds, zero_division=0),
        "F1-Score": f1_score(df_test['label'], test_preds, zero_division=0)
    }
    return metrics, confusion_matrix(df_test['label'], test_preds)

def main():
    print("="*60)
    print("BI-ENCODER EXPERIMENT (Zero-Shot vs Fine-Tuned)")
    print("="*60)
    
    # 1. Φόρτωση Δεδομένων
    print(f"\n1. Loading Datasets (Train frac: {TRAIN_FRACTION*100}%)...")
    train_df = load_and_format_data(DATASET_PATH, "train", frac=TRAIN_FRACTION)
    valid_df = load_and_format_data(DATASET_PATH, "valid")
    test_df  = load_and_format_data(DATASET_PATH, "test")
    
    print(f"   Train samples (for Fine-Tuning): {len(train_df)}")
    print(f"   Valid samples (for Threshold):   {len(valid_df)}")
    print(f"   Test samples  (for Final Score): {len(test_df)}")
    
    # 2. Φόρτωση Μοντέλου
    print(f"\n2. Loading Pre-trained Bi-Encoder ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 3. ZERO-SHOT ΑΞΙΟΛΟΓΗΣΗ
    print("\n3. ZERO-SHOT EVALUATION (Off-the-shelf)")
    zero_metrics, zero_cm = evaluate_model(model, valid_df, test_df)
    print(f"   Best Threshold (from Valid): {zero_metrics['Threshold']:.2f}")
    print(f"   Test F1-Score: {zero_metrics['F1-Score']:.4f}")
    print(f"   Test Precision: {zero_metrics['Precision']:.4f} | Recall: {zero_metrics['Recall']:.4f}")
    
    # 4. FINE-TUNING (Contrastive Loss)
    print("\n4. FINE-TUNING BI-ENCODER (Contrastive Loss)...")
    train_examples = []
    for _, row in train_df.iterrows():
        # To ContrastiveLoss δέχεται label 1.0 για ταίριασμα, 0.0 για διαφορά
        train_examples.append(InputExample(texts=[row['text1'], row['text2']], label=float(row['label'])))
        
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.ContrastiveLoss(model=model)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=int(len(train_dataloader) * 0.1), # 10% warmup
        show_progress_bar=True
    )
    
    # 5. FINE-TUNED ΑΞΙΟΛΟΓΗΣΗ
    print("\n5. FINE-TUNED EVALUATION (After 30% Training)")
    tuned_metrics, tuned_cm = evaluate_model(model, valid_df, test_df)
    print(f"   Best Threshold (from Valid): {tuned_metrics['Threshold']:.2f}")
    print(f"   Test F1-Score: {tuned_metrics['F1-Score']:.4f}")
    print(f"   Test Precision: {tuned_metrics['Precision']:.4f} | Recall: {tuned_metrics['Recall']:.4f}")
    
    # ---------------------------------------------------------
    # ΔΗΜΙΟΥΡΓΙΑ ΤΟΥ REPORT ΚΑΙ ΑΠΟΘΗΚΕΥΣΗ ΣΕ ΑΡΧΕΙΟ TXT
    # ---------------------------------------------------------
    report_content = (
        "="*60 + "\n"
        "SUMMARY OF RESULTS (WALMART-AMAZON Test Set)\n"
        + "="*60 + "\n"
        f"{'Method':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}\n"
        + "-" * 55 + "\n"
        f"{'Zero-Shot':<15} | {zero_metrics['Precision']:<10.4f} | {zero_metrics['Recall']:<10.4f} | {zero_metrics['F1-Score']:<10.4f}\n"
        f"{'Fine-Tuned (30%)':<15} | {tuned_metrics['Precision']:<10.4f} | {tuned_metrics['Recall']:<10.4f} | {tuned_metrics['F1-Score']:<10.4f}\n"
        + "="*60 + "\n\n"
        "--- DETAILED METRICS ---\n"
        f"Zero-Shot Optimal Threshold (from Valid): {zero_metrics['Threshold']:.2f}\n"
        f"Fine-Tuned Optimal Threshold (from Valid): {tuned_metrics['Threshold']:.2f}\n\n"
        "--- CONFUSION MATRICES (Test Set) ---\n"
        "Zero-Shot Confusion Matrix:\n"
        f"{zero_cm}\n\n"
        "Fine-Tuned Confusion Matrix:\n"
        f"{tuned_cm}\n"
    )

    print("\n" + report_content)

    report_filename = "bi_encoder_results.txt"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"\nΤα αποτελέσματα αποθηκεύτηκαν επιτυχώς στο αρχείο: {report_filename}\n")

if __name__ == "__main__":
    # Σταθεροποίηση seeds για αναπαραγωγιμότητα
    torch.manual_seed(42)
    np.random.seed(42)
    main()