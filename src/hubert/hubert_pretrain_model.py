import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import HubertForSequenceClassification, HubertConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, F1Score
from typing import Optional, Dict, Any
import torch.nn.functional as F
import math


class HubertClassifierLightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module for HuBERT-based audio classification.
    
    Args:
        model_name: HuggingFace model name or path (e.g., 'facebook/hubert-base-ls960')
        num_labels: Number of classification labels
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        warmup_steps: Number of warmup steps for learning rate scheduler
        max_epochs: Maximum number of training epochs (for scheduler)
        freeze_feature_encoder: Whether to freeze the feature encoder layers
        freeze_layers: Number of transformer layers to freeze (0 = none)
    """
    
    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        num_labels: int = 2,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_epochs: int = 10,
        freeze_feature_encoder: bool = False,
        freeze_layers: int = 0,
        freeze_base_model: bool = True,
        max_steps: int = 1000,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained HuBERT model
        self.model = HubertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

        # Freeze layers if specified
        if freeze_feature_encoder:
            for param in self.model.hubert.feature_extractor.parameters():
                param.requires_grad = False
                
        if freeze_layers > 0:
            for layer in self.model.hubert.encoder.layers[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        if freeze_base_model:
            self.model.freeze_base_model()
            
        self.model.train()
        
        
        # Initialize metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_labels)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_labels)
        
        self.train_f1 = F1Score(task="multiclass", num_classes=num_labels, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_labels, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_labels, average="macro")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()


        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps


    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, input_dim, seq_len)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        # Get features
        logits = self.model(x).logits  # (batch, seq_len, hidden_dim)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, stage="train")
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, stage="val")
        
        return loss
    

    def _shared_step(self, batch, batch_idx, stage: str):
        """Shared step for training, validation, and testing."""
        if stage=="train":
            self.model.train()
        else:
            self.model.eval()
        inputs, labels = batch
        logits = self(inputs)
        
        # Convert one-hot to class indices if needed
        if labels.dim() > 1 and labels.size(1) > 1:
            labels = torch.argmax(labels, dim=1)
        
        loss = F.cross_entropy(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # Update metrics
        if stage == "train":
            self.train_accuracy(preds, labels)
            self.train_f1(preds, labels)
            self.log(f"{stage}_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_f1", self.train_f1, on_step=True, on_epoch=True)
        elif stage == "val":
            self.val_accuracy(preds, labels)
            self.val_f1(preds, labels)
            self.log(f"{stage}_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_f1", self.val_f1, on_step=False, on_epoch=True)
        elif stage == "test":
            all_same = torch.all(labels == labels[0])
            assert all_same, "All samples in the test batch should belong to the same class."

            self.test_accuracy(preds, labels)
            self.test_f1(preds, labels)
            self.log(f"{stage}_acc", self.test_accuracy, on_step=True, on_epoch=False)
            self.log(f"{stage}_f1", self.test_f1, on_step=True, on_epoch=False)
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
            self.log(f"{stage}_label", labels[0], on_step=True, on_epoch=False, prog_bar=True)
        
        # Log loss
        if not stage=="test":
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="test")
    
    def configure_optimizers(self):
        # # Separate parameters for weight decay
        # no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        # optimizer_grouped_parameters = [
        #     {
        #         'params': [p for n, p in self.named_parameters() 
        #                   if not any(nd in n for nd in no_decay)],
        #         'weight_decay': self.weight_decay,
        #     },
        #     {
        #         'params': [p for n, p in self.named_parameters() 
        #                   if any(nd in n for nd in no_decay)],
        #         'weight_decay': 0.0,
        #     },
        # ]
        
        # optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        # # Warmup + cosine annealing scheduler
        # def lr_lambda(current_step):
        #     if current_step < self.warmup_steps:
        #         return float(current_step) / float(max(1, self.warmup_steps))
        #     progress = float(current_step - self.warmup_steps) / float(
        #         max(1, self.max_steps - self.warmup_steps)
        #     )
        #     return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        # scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
        #     'interval': 'step',
        #     'frequency': 1,
        # }

        optimizer = AdamW(params = [p for n, p in self.named_parameters()],lr=self.learning_rate)
        
        return [optimizer]#, [scheduler]
    


if __name__ == "__main__":
    # Example instantiation
    model = HubertClassifierLightningModule(
        model_name="facebook/hubert-base-ls960",
        num_labels=10,  # e.g., 10 audio classes
        learning_rate=5e-5,
        weight_decay=0.01,
        max_epochs=10,
        freeze_feature_encoder=False,
        freeze_layers=0
    )
    
    print("Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")