import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

# ==========================================
# 1. Define the Architecture and Logic
# ==========================================
class LitClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        # A simple model: Input layer -> Activation -> Output layer
        self.layer1 = nn.Linear(20, 64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, 2)

        # Save hyperparameters (like learning rate) so they show up in TensorBoard
        self.save_hyperparameters(logger=True)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        
        # Log to TensorBoard automatically on every step and epoch
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        
        # Track accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log validation metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # Using Adam optimizer with a learning rate of 0.001
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# ==========================================
# 2. Setup Dummy Data Generators
# ==========================================
def get_toy_dataloaders():
    # Create random feature vectors (1000 samples, 20 features each)
    x_train = torch.randn(1000, 20)
    # Create random binary labels (0 or 1)
    y_train = torch.randint(0, 2, (1000,))
    
    x_val = torch.randn(200, 20)
    y_val = torch.randint(0, 2, (200,))

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


# ==========================================
# 3. Execution Main Block
# ==========================================
if __name__ == "__main__":
    # Ensure reproducibility
    L.seed_everything(42)

    # Initialize data loaders
    train_loader, val_loader = get_toy_dataloaders()

    # Initialize the model
    model = LitClassifier()

    # Explicitly configure the TensorBoard Logger
    tb_logger = TensorBoardLogger(save_dir="tb_logs", name="my_classification_model")

    # Pass the logger to the Trainer boilerplate
    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",  # Automatically uses GPU if available, else CPU
        logger=tb_logger,
        log_every_n_steps=10  # Log frequently since our dummy dataset is tiny
    )

    # Run the training and validation loops
    print("--- Starting Training Process ---")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("--- Training Completed Successfully ---")