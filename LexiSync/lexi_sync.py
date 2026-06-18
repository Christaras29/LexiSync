import os
import re
import torch
import pandas as pd
import numpy as np
from typing import List, Tuple
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import random
from trainer import WeightedTrainer


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class LexiSync:
    def __init__(self, model_name: str = "microsoft/deberta-v3-base", output_dir: str = "./output"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.datasets = {}      # Αποθήκευση datasets
        self.weights = {}       # Αποθήκευση βαρών
        self.specs = {}         # Αποθήκευση δομής (στήλες)
        
        print(f"Initializing {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            
        # Φόρτωση Μοντέλου (Explicit float32 για MPS stability)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, torch_dtype=torch.float32
        )
        
        # Λίστα ειδικών tokens 
        self.special_tokens = set()

    def _clean_text(self, text):
        """Καθαρισμός κειμένου από θόρυβο."""
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s\.\-]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _calculate_weights(self, dataset):
        """Υπολογισμός βαρών για handling class imbalance."""
        labels = [x['label'] for x in dataset]
        pos = sum(labels)
        neg = len(labels) - pos
        if pos == 0 or neg == 0: return [1.0, 1.0] 
        w0 = (pos + neg) / (2 * neg)
        w1 = ((pos + neg) / (2 * pos)) * 0.5
        return [w0, w1]

    def add_dataset(self, name: str, data_path: str, columns_spec: List[Tuple[str, str]]):
        """
        Φόρτωση ενός dataset.
        name: Όνομα αναφοράς (π.χ. 'amazon')
        data_path: Φάκελος που περιέχει τα source.csv, target.csv, train.csv, test.csv
        columns_spec: Λίστα με (Tag, ColumnName) π.χ. [('[T]', 'title'), ('[P]', 'price')]
        """
        print(f"Loading dataset: {name} from {data_path}...")
        
        # Προσθήκη των tags στα special tokens
        for tag, _ in columns_spec:
            self.special_tokens.add(tag)
        
        def read_csv(f): return pd.read_csv(os.path.join(data_path, f))
        
        try:
            src = read_csv("source.csv")
            tgt = read_csv("target.csv")
            train_df = read_csv("train.csv")
            valid_df = read_csv("valid.csv")   # ⭐ ΝΕΟ
            test_df = read_csv("test.csv")
        except FileNotFoundError as e:
            print(f"Error loading {name}: {e}")
            return

        self.specs[name] = columns_spec
        self.datasets[name] = {}

        # ⭐ Επεξεργασία Train, Valid και Test
        for split_name, split_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
            merged = split_df.merge(src, left_on="ltable_id", right_on="id").merge(tgt, left_on="rtable_id", right_on="id", suffixes=('_l', '_r'))
            
            # Στο train χρησιμοποιούμε όλα τα δεδομένα (sample frac=1 για shuffle)
            if split_name == "train":
                merged = merged.sample(frac=1, random_state=42)

            texts = []
            for _, r in merged.iterrows(): # Δημιουργία text pairs με tags
                l_parts, r_parts = [], []
                for tag, col in columns_spec:
                    val_l = self._clean_text(r[col+'_l'] if col+'_l' in r else '')
                    val_r = self._clean_text(r[col+'_r'] if col+'_r' in r else '')
                    l_parts.append(f"{tag} {val_l}")
                    r_parts.append(f"{tag} {val_r}")
                texts.append({"text1": " ".join(l_parts), "text2": " ".join(r_parts), "label": int(r['label'])})
            
            ds = Dataset.from_list(texts)
            
            # Tokenization
            def tok(b): return self.tokenizer(b["text1"], b["text2"], truncation=True, max_length=128)
            self.datasets[name][split_name] = ds.map(tok, batched=True)
        
        # Υπολογισμός βαρών για το train set
        self.weights[name] = self._calculate_weights(self.datasets[name]["train"])
        print(f"   -> Loaded {len(self.datasets[name]['train'])} train, "f"{len(self.datasets[name]['valid'])} valid, "f"{len(self.datasets[name]['test'])} test examples.")
        print(f"   -> Class Weights: Neg={self.weights[name][0]:.2f}, Pos={self.weights[name][1]:.2f}")

    def _update_tokenizer(self):
        """Ενημέρωση του tokenizer με τα νέα special tokens."""
        if self.special_tokens:
            self.tokenizer.add_special_tokens({'additional_special_tokens': list(self.special_tokens)})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def train_source(self, source_name: str, epochs: int = 5, batch_size: int = 12):
        """Βήμα 1: Εκπαίδευση στο Source Domain."""
        if source_name not in self.datasets:
            raise ValueError(f"Dataset {source_name} not found.")
        
        self._update_tokenizer()
        
        print(f"\n--- Phase 1: Training on Source ({source_name}) ---")
        
        args = TrainingArguments(
            output_dir=f"{self.output_dir}/{source_name}",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=2e-5,
            save_strategy="no",
            report_to="none",
            weight_decay=0.01
        )
        
        self.trainer = WeightedTrainer(
            model=self.model,
            args=args,
            train_dataset=self.datasets[source_name]["train"],
            data_collator=DataCollatorWithPadding(self.tokenizer),
            class_weights=self.weights[source_name]
        )
        
        self.trainer.train()
        print("Source training complete.")

    def adapt_to_targets(self, source_name: str, target_names: List[str], replay_samples: int = 1000, epochs: int = 6, batch_size: int = 12):
        """Βήμα 2: Transfer Learning στα Target Domains με Anti-Forgetting."""
        print(f"\n--- Phase 2: Adaptation to {target_names} ---")
        
        # 1. Replay Buffer από Source, εδώ κάνουμε shuffle και παίρνουμε δείγματα
        source_ds = self.datasets[source_name]["train"]
        replay_ds = source_ds.shuffle(seed=42).select(range(min(replay_samples, len(source_ds))))
        
        # 2. Συλλογή Target Datasets, δημιουργία joint dataset, δηλαδη replay + targets
        datasets_to_mix = [replay_ds]
        for t_name in target_names:
            if t_name in self.datasets:
                datasets_to_mix.append(self.datasets[t_name]["train"])
            else:
                print(f"Warning: Target {t_name} not found. Skipping.")
        
        joint_ds = concatenate_datasets(datasets_to_mix).shuffle(seed=42) # Ανακάτεμα των δεδομένων
        joint_weights = self._calculate_weights(joint_ds)
        
        print(f"Joint Dataset Size: {len(joint_ds)} | Weights: {joint_weights}")

        args = TrainingArguments(
            output_dir=f"{self.output_dir}/joint",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=1e-5, #
            save_strategy="no",
            report_to="none",
            weight_decay=0.01
        )
        
        self.trainer = WeightedTrainer( 
            model=self.model,
            args=args,
            train_dataset=joint_ds,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            class_weights=joint_weights
        )
        
        self.trainer.train()
        print("Adaptation complete.")

    def generate_report(self, filename="final_report.txt"):
        """
        Δημιουργία report για ΟΛΑ τα φορτωμένα datasets.
        Threshold tuning στο validation set, final evaluation στο test set.
        """
        print(f"\nGenerating report to {filename}...")
        
        with open(filename, "w") as f:
            header = f"ENTITY RESOLUTION REPORT (Model: {self.model_name})\n" + "="*60 + "\n"
            f.write(header)
            print(header)

            for name in self.datasets:
                # ============================================================
                # ΒΗΜΑ 1: Threshold tuning στο VALIDATION set
                # ============================================================
                val_output = self.trainer.predict(self.datasets[name]["valid"])
                val_probs = torch.nn.functional.softmax(
                    torch.from_numpy(val_output.predictions), dim=-1
                )[:, 1].numpy()
                val_labels = val_output.label_ids

                best_thresh, best_val_f1 = 0.5, 0
                for t in np.arange(0.2, 0.95, 0.05):
                    t_f1 = f1_score(val_labels, (val_probs >= t).astype(int))
                    if t_f1 > best_val_f1:
                        best_val_f1, best_thresh = t_f1, t

                # ============================================================
                # ΒΗΜΑ 2: Final evaluation στο TEST set με το σταθερό threshold
                # ============================================================
                test_output = self.trainer.predict(self.datasets[name]["test"])
                test_probs = torch.nn.functional.softmax(
                    torch.from_numpy(test_output.predictions), dim=-1
                )[:, 1].numpy()
                test_labels = test_output.label_ids

                y_hat = (test_probs >= best_thresh).astype(int)

                # ============================================================
                # ΒΗΜΑ 3: Reporting
                # ============================================================
                res = f"\n>>> DATASET: {name.upper()}\n"
                res += f"Optimal Threshold (tuned on validation): {best_thresh:.2f}\n"
                res += f"Validation F1 (during tuning): {best_val_f1:.4f}\n"
                res += f"\nTest Confusion Matrix:\n{confusion_matrix(test_labels, y_hat)}\n"
                res += f"\nTest Classification Report:\n{classification_report(test_labels, y_hat, digits=4)}\n"
                res += "-" * 60 + "\n"

                print(res)
                f.write(res)

        print(f"Done. Report saved.")

    def save_final_model(self, path: str = "./saved_model"):
        """Αποθήκευση του τελικού μοντέλου και του tokenizer."""
        print(f"Saving final model to {path}...")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

if __name__ == "__main__":
    models_to_test = [
        "microsoft/deberta-v3-base",
        "sentence-transformers/all-MiniLM-L6-v2",
        "Snowflake/snowflake-arctic-embed-xs",
        "BAAI/bge-small-en-v1.5",
        "thenlper/gte-small",
        "intfloat/e5-small-v2"
    ]

    base_dir = "./data"
    
    for model_name in models_to_test:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        print(f"\n{'='*50}")
        print(f"STARTING EXPERIMENT WITH: {model_name}")
        print(f"{'='*50}")

        safe_model_name = model_name.replace("/", "_")
        current_output_dir = f"./output_{safe_model_name}"

        # 1. Αρχικοποίηση
        matcher = LexiSync(model_name=model_name, output_dir=current_output_dir)

        # 2. Φόρτωση Datasets 
        matcher.add_dataset(
            name="dblp", 
            data_path=f"{base_dir}/DBLP-Scholar", 
            columns_spec=[("[T]", "title"), ("[A]", "authors"), ("[V]", "venue"), ("[Y]", "year")]
        )
        matcher.add_dataset(
            name="amazon", 
            data_path=f"{base_dir}/Amazon-Google", 
            columns_spec=[("[T]", "title"), ("[M]", "manufacturer"), ("[P]", "price")]
        )
        matcher.add_dataset(
            name="walmart", 
            data_path=f"{base_dir}/Walmart-Amazon", 
            columns_spec=[("[T]", "title"), ("[N]", "modelno"), ("[B]", "brand"), ("[C]", "category"), ("[P]", "price")]
        )
        
        # 3. Εκπαίδευση & Προσαρμογή
        matcher.train_source(source_name="dblp", epochs=5)
        matcher.adapt_to_targets(source_name="dblp", target_names=["amazon", "walmart"], replay_samples=1200, epochs=7)

        # 4. Αξιολόγηση - Αποθήκευση σε διαφορετικό αρχείο ανά μοντέλο
        matcher.generate_report(f"results_{safe_model_name}.txt")

        # Απελευθέρωση μνήμης για το επόμενο μοντέλο
        del matcher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nCompleted successfully.")