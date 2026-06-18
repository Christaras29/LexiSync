import os
import torch
from transformers import Trainer
from torch import nn


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 1. Helper: Weighted Trainer
class WeightedTrainer(Trainer):
    """Custom Trainer που διαχειρίζεται το Weighted Loss για ανισορροπία κλάσεων."""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None: # Αν κάνει λάθος match τότε το τιμωρεί με βάρος ίσο με το weight
            weights = self.class_weights.to(logits.device).to(logits.dtype)
            loss_fct = nn.CrossEntropyLoss(weight=weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1)) 
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
        return (loss, outputs) if return_outputs else loss