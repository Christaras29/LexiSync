import pandas as pd
import torch
import faiss
import time
from sentence_transformers import SentenceTransformer

class ANNSBlocker:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Αρχικοποίηση Blocker.
        """
        print(f"Φόρτωση Bi-Encoder για Blocking: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        if torch.cuda.is_available():
            self.encoder = self.encoder.to("cuda")

    def create_candidates(self, source_csv, target_csv, top_k=50, threshold=0.5):
        """
        Δημιουργία υποψήφιων ζευγών.
        """
        # 1. Φόρτωση και προετοιμασία δεδομένων
        df_src = pd.read_csv(source_csv).fillna("")
        df_tgt = pd.read_csv(target_csv).fillna("")
        
        # Σειριοποίηση
        src_texts = df_src.astype(str).apply(lambda x: " ".join(x), axis=1).tolist()
        tgt_texts = df_tgt.astype(str).apply(lambda x: " ".join(x), axis=1).tolist()

        print(f"Embedding {len(src_texts)} εγγραφών από την Πηγή...")
        start_time = time.time()
        src_embeddings = self.encoder.encode(src_texts, convert_to_numpy=True, show_progress_bar=True)
        tgt_embeddings = self.encoder.encode(tgt_texts, convert_to_numpy=True, show_progress_bar=True)
        
        # 2. Κατασκευή ANNS Index με FAISS (Inner Product για Cosine Similarity)
        faiss.omp_set_num_threads(1)
        print("Κατασκευή FAISS Index...")
        faiss.normalize_L2(src_embeddings)
        faiss.normalize_L2(tgt_embeddings)
        
        dimension = src_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension) # Απλή και ακριβής αναζήτηση
        index.add(tgt_embeddings)

        # 3. Αναζήτηση των top-k γειτόνων
        print(f"Αναζήτηση των {top_k} επικρατέστερων υποψηφίων για κάθε εγγραφή...")
        distances, indices = index.search(src_embeddings, top_k)
        
        # 4. Φιλτράρισμα και δημιουργία λίστας ζευγών
        candidate_pairs = []
        for i, neighbor_indices in enumerate(indices):
            for j, tgt_idx in enumerate(neighbor_indices):
                score = distances[i][j]
                if score >= threshold:
                    candidate_pairs.append({
                        "ltable_id": df_src.iloc[i]['id'],
                        "rtable_id": df_tgt.iloc[tgt_idx]['id'],
                        "blocking_score": float(score)
                    })
        
        end_time = time.time()
        print(f"Ολοκληρώθηκε! Βρέθηκαν {len(candidate_pairs)} υποψήφια ζεύγη σε {end_time-start_time:.2f} δευτερόλεπτα.")
        
        # Αποθήκευση μεγεθών για evaluation
        self.n_source = len(df_src)
        self.n_target = len(df_tgt)
        
        return pd.DataFrame(candidate_pairs)

    def evaluate(self, candidates_df, train_csv, valid_csv, test_csv):
        """
        Σύντομη αξιολόγηση: σύγκριση candidates με ground truth matches.
        Το ground truth συγκεντρώνεται από train + valid + test.
        """
        # Φόρτωση ground truth από όλα τα splits (train, valid, test)
        df_gt = pd.concat([
            pd.read_csv(train_csv), 
            pd.read_csv(valid_csv),
            pd.read_csv(test_csv)
        ], ignore_index=True)
        
        true_matches = df_gt[df_gt['label'] == 1]
        
        # Auto-detect ID columns (ltable_id/rtable_id ή LTB_ID/RTB_ID)
        left_col = 'ltable_id' if 'ltable_id' in df_gt.columns else 'LTB_ID'
        right_col = 'rtable_id' if 'rtable_id' in df_gt.columns else 'RTB_ID'
        
        true_pairs = set(zip(true_matches[left_col], true_matches[right_col]))
        candidate_pairs = set(zip(candidates_df['ltable_id'], candidates_df['rtable_id']))
        retrieved = candidate_pairs & true_pairs
        
        # Μετρικές
        pc = len(retrieved) / len(true_pairs) if true_pairs else 0
        pq = len(retrieved) / len(candidate_pairs) if candidate_pairs else 0
        rr = 1 - (len(candidate_pairs) / (self.n_source * self.n_target))
        
        report = ""
        report += "=" * 50 + "\n"
        report += "BLOCKER EVALUATION\n"
        report += "=" * 50 + "\n"
        report += f"True matches (train+valid+test):  {len(true_pairs)}\n" 
        report += f"Candidates:             {len(candidate_pairs):,}\n"
        report += f"Matches retrieved:      {len(retrieved)} / {len(true_pairs)}\n"
        report += f"Matches missed:         {len(true_pairs) - len(retrieved)}\n"
        report += f"Pairs Completeness:     {pc:.4f} ({pc*100:.2f}%)\n"
        report += f"Pairs Quality:          {pq:.4f} ({pq*100:.2f}%)\n"
        report += f"Reduction Ratio:        {rr:.4f} ({rr*100:.2f}%)\n"
        report += "=" * 50 + "\n"
        print(report)
        return report


if __name__ == "__main__":
    # Παράδειγμα χρήσης για το Walmart
    blocker = ANNSBlocker()
    
    results = blocker.create_candidates(
        source_csv="./data/Walmart-Amazon/source.csv",
        target_csv="./data/Walmart-Amazon/target.csv",
        top_k=10,
        threshold=0.5
    )
    
    results.to_csv("data/candidates.csv", index=False)
    print("Τα υποψήφια ζεύγη αποθηκεύτηκαν στο 'data/candidates.csv'")
    
    # Σύγκριση με ground truth από όλα τα splits
    report = blocker.evaluate(
        candidates_df=results,
        train_csv="./data/Walmart-Amazon/train.csv",
        valid_csv="./data/Walmart-Amazon/valid.csv",
        test_csv="./data/Walmart-Amazon/test.csv"
    )
    report_path = "data/blocker_evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Η αναφορά αξιολόγησης αποθηκεύτηκε στο '{report_path}'")