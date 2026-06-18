import os
import torch
import numpy as np
import random
import warnings
import json
from sklearn.metrics import f1_score
from datasets import Dataset
from lexi_sync import LexiSync 

warnings.filterwarnings("ignore")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class AblationStudy(LexiSync):
    """
    Subclass της LexiSync για ablation study.
    
    Stratified sub-sampling: διατηρεί την ίδια αναλογία θετικών/αρνητικών
    με το αρχικό dataset, διασφαλίζοντας στατιστική αντιπροσωπευτικότητα.
    """
    
    def __init__(self, model_name="microsoft/deberta-v3-base", 
                 output_dir="./output", w1_factor=0.5, sample_fraction=0.5):
        super().__init__(model_name=model_name, output_dir=output_dir)
        
        self.w1_factor = w1_factor
        self.sample_fraction = sample_fraction
        
        print(f"--> [Fast Mode] w1_factor = {self.w1_factor}, "
              f"sample_fraction = {self.sample_fraction} (stratified)")

    # [OVERRIDE] Δυναμικός υπολογισμός βαρών με το w1_factor του πειράματος
    def _calculate_weights(self, dataset):
        labels = [x['label'] for x in dataset]
        pos = sum(labels)
        neg = len(labels) - pos
        if pos == 0 or neg == 0: 
            return [1.0, 1.0] 
        w0 = (pos + neg) / (2 * neg)
        w1 = ((pos + neg) / (2 * pos)) * self.w1_factor
        return [w0, w1]

    # Stratified sub-sampling που διατηρεί την αναλογία κλάσεων
    def _stratified_subsample(self, dataset, fraction, seed=42):
        """
        Παίρνει ένα subset του dataset διατηρώντας την ίδια αναλογία 
        θετικών/αρνητικών δειγμάτων (stratified sampling).
        """
        # Διαχωρισμός σε positives και negatives
        pos_indices = [i for i, x in enumerate(dataset) if x['label'] == 1]
        neg_indices = [i for i, x in enumerate(dataset) if x['label'] == 0]
        
        # Random shuffle με seed για reproducibility
        rng = random.Random(seed)
        rng.shuffle(pos_indices)
        rng.shuffle(neg_indices)
        
        # Επιλογή ίδιου ποσοστού από κάθε κλάση
        n_pos = max(1, int(len(pos_indices) * fraction))
        n_neg = max(1, int(len(neg_indices) * fraction))
        
        selected_indices = pos_indices[:n_pos] + neg_indices[:n_neg]
        rng.shuffle(selected_indices)  # Ανακάτεμα του τελικού subset
        
        return dataset.select(selected_indices), n_pos, n_neg

    # [OVERRIDE] add_dataset με stratified sub-sampling
    def add_dataset(self, name, data_path, columns_spec):
        # Καλούμε την αρχική add_dataset
        super().add_dataset(name, data_path, columns_spec)
        
        # Stratified sub-sampling στο training set
        if name in self.datasets and self.sample_fraction < 1.0:
            train_ds = self.datasets[name]["train"]
            
            # Original ratio
            orig_pos = sum(1 for x in train_ds if x['label'] == 1)
            orig_neg = len(train_ds) - orig_pos
            orig_ratio = orig_neg / orig_pos if orig_pos > 0 else 0
            
            # Stratified sub-sampling
            subsampled, n_pos, n_neg = self._stratified_subsample(
                train_ds, self.sample_fraction, seed=42
            )
            self.datasets[name]["train"] = subsampled
            new_ratio = n_neg / n_pos if n_pos > 0 else 0
            
            # Επανυπολογισμός βαρών για το reduced dataset
            self.weights[name] = self._calculate_weights(self.datasets[name]["train"])
            
            print(f"   -> Stratified sub-sample: {n_pos} pos + {n_neg} neg = {n_pos+n_neg} total")
            print(f"   -> Original ratio 1:{orig_ratio:.2f} | New ratio 1:{new_ratio:.2f}")


    # [ΝΕΑ ΜΕΘΟΔΟΣ] Επιστρέφει τα scores
    def get_eval_metrics(self):
        """
        Threshold tuning στο VALIDATION, final evaluation στο TEST.
        Επιστρέφει F1-scores.
        """
        metrics = {}
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
                t_f1 = f1_score(val_labels, (val_probs >= t).astype(int), zero_division=0)
                if t_f1 > best_val_f1:
                    best_val_f1, best_thresh = t_f1, t

            # ============================================================
            # ΒΗΜΑ 2: Final evaluation στο TEST με το threshold του valid
            # ============================================================
            test_output = self.trainer.predict(self.datasets[name]["test"])
            test_probs = torch.nn.functional.softmax(
                torch.from_numpy(test_output.predictions), dim=-1
            )[:, 1].numpy()
            test_labels = test_output.label_ids

            test_f1 = f1_score(
                test_labels, 
                (test_probs >= best_thresh).astype(int), 
                zero_division=0
            )

            metrics[name] = {
                "F1-Score": round(test_f1, 4),              
                "Optimal Threshold": round(best_thresh, 2),
                "Validation F1": round(best_val_f1, 4)   
            }
        return metrics


# =====================================================================
# 3. ΕΚΤΕΛΕΣΗ ΠΕΙΡΑΜΑΤΟΣ
# =====================================================================
if __name__ == "__main__":
    model_name = "microsoft/deberta-v3-base"
    base_dir = "./data"
    
    # 3 multipliers
    w1_grid = [0.3, 0.5, 1.0]
    
    datasets_config = {
        "dblp": [("[T]", "title"), ("[A]", "authors"), 
                 ("[V]", "venue"), ("[Y]", "year")],
        "amazon": [("[T]", "title"), ("[M]", "manufacturer"), ("[P]", "price")],
        "walmart": [("[T]", "title"), ("[N]", "modelno"), ("[B]", "brand"), 
                    ("[C]", "category"), ("[P]", "price")]
    }
    
    all_results = {w: {} for w in w1_grid}

    print(f"{'='*60}")
    print(f"ABLATION STUDY — Stratified Sub-sampling")
    print(f"{'='*60}")

    for w1 in w1_grid:
        print(f"\n{'*'*40}")
        print(f"---> ΕΚΠΑΙΔΕΥΣΗ ΜΕ w1 = {w1} <---")
        print(f"{'*'*40}")
        
        # Reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available(): 
            torch.cuda.manual_seed_all(42)
            
        current_output_dir = f"./output_gridsearch_w1_{w1}"

        # Subclass με stratified sub-sampling 50%
        matcher = AblationStudy(
            model_name=model_name, 
            output_dir=current_output_dir, 
            w1_factor=w1,
            sample_fraction=0.5  # 50% με διατήρηση αναλογίας
        )

        for ds_name, config in datasets_config.items():
            path = f"{base_dir}/"
            if ds_name == "dblp":
                path += "DBLP-Scholar"
            elif ds_name == "amazon":
                path += "Amazon-Google"
            else:
                path += "Walmart-Amazon"
            matcher.add_dataset(ds_name, path, config)
        
        # ίδια μεθοδολογία εκπαίδευσης
        matcher.train_source(source_name="dblp", epochs=3)
        matcher.adapt_to_targets(
            source_name="dblp", 
            target_names=["amazon", "walmart"], 
            replay_samples=600,  
            epochs=3
        )

        all_results[w1] = matcher.get_eval_metrics()

        del matcher
        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()

    # Αποτελέσματα
    datasets = ["dblp", "amazon", "walmart"]
    
    print(f"\n\n{'='*60}")
    print(f"ΑΝΑΛΥΤΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ (ΑΝΑ ΣΥΝΟΛΟ)")
    print(f"{'='*60}")
    for ds_name in datasets:
        print(f"\n>>> {ds_name.upper()} <<<")
        for w1 in w1_grid:
            score = all_results[w1][ds_name]["F1-Score"]
            thresh = all_results[w1][ds_name]["Optimal Threshold"]
            print(f"  w1 = {w1}  -->  F1: {score:.4f}  (Threshold: {thresh:.2f})")
            
    print(f"\n\n{'='*60}")
    print(f"ΚΑΘΟΛΙΚΟ w1 (MACRO-AVERAGE)")
    print(f"{'='*60}")
    
    global_averages = {}
    for w1 in w1_grid:
        avg_f1 = sum(all_results[w1][ds]["F1-Score"] for ds in datasets) / len(datasets)
        global_averages[w1] = avg_f1
        
    best_global_w1 = max(global_averages, key=global_averages.get)

    for w1 in w1_grid:
        is_best = "  <-- ΒΕΛΤΙΣΤΟ" if w1 == best_global_w1 else ""
        print(f"  w1 = {w1}  -->  Μέσος F1: {global_averages[w1]:.4f}{is_best}")
    
    # Αποθήκευση
    with open("ablation.json", "w") as f:
        json.dump({
            "methodology": "full fine-tuning with 50% stratified sub-sampling",
            "w1_grid": w1_grid,
            "results": {str(w): all_results[w] for w in w1_grid},
            "global_averages": {str(w): global_averages[w] for w in w1_grid},
            "best_w1": best_global_w1,
        }, f, indent=2)
    
    print(f"\nResults saved to ablation.json")
    print(f"\nDone.")