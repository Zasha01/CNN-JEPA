# Small sleep to prevent UI lag
import os
from time import sleep

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm

import wandb
from src.models.base_classifier import BaseClassifier
from src.utils.wandb_utils import log_confusion_matrix, log_model_summary


class TactNetEncoder(nn.Module):
    """
    Encoder for TactNet model.
    """

    def __init__(self, in_channels=1, out_channels=128):
        """
        Initializes the encoder.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels, 32, kernel_size=(15, 5), stride=(1, 1), padding=(7, 2)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(10, 1), stride=(10, 1)),
            nn.Conv2d(32, 64, kernel_size=(15, 5), stride=(1, 1), padding=(7, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(10, 1), stride=(10, 1)),
            nn.Conv2d(
                64, out_channels, kernel_size=(15, 5), stride=(1, 1), padding=(7, 2)
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(10, 1), stride=(10, 1)),
        )

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Encoded output.
        """
        return self.encoder(x)

    def initialize_weights(self):
        """
        Initialize weights using Kaiming initialization for better training stability.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)


class TactNet2(BaseClassifier):
    """
    Improved TactNet2 model with stability enhancements.
    """

    def __init__(
        self, num_classes=36, lr=1e-4, weight_decay=1e-4, dropout_p=0.35, device=None
    ):
        """
        Initializes the improved TactNet2 model.

        Args:
            num_classes (int): Number of output classes.
            lr (float): Learning rate (reduced for better stability).
            weight_decay (float): Weight decay (increased for better regularization).
            dropout_p (float): Dropout probability (increased for better regularization).
            device (str): Device to use ("cuda" or "cpu").
        """
        super().__init__()

        self.weight_decay = weight_decay
        self.lr = lr

        # Improved classifier with more regularization
        self.encoder = TactNetEncoder()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128 * 1 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

        # Initialize weights properly
        self.initialize_weights()
        self.to(device)
        self.scheduler = None  # Will be set in set_optimizer
        self.set_optimizer()

    def initialize_weights(self):
        """
        Initialize weights using Kaiming initialization for better training stability.
        """
        self.encoder.initialize_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor of shape (B, 1, H, W).

        Returns:
            Tensor: Output logits of shape (B, num_classes).
        """
        encoded_x = self.encoder(x)
        out = self.classifier(encoded_x)
        return out

    def set_optimizer(self):
        """
        Sets the optimizer and learning rate scheduler for the model.
        """
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),  # Default betas for AdamW
            eps=1e-8,  # Default epsilon for numerical stability
        )

        # Use StepLR scheduler to reduce LR every 20 epochs
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=20,  # Reduce learning rate every 20 epochs
            gamma=0.5,  # Reduce learning rate by a factor of 10 each time
        )
        self.iterations = 0

    def train_model(
        self,
        train_loader,
        val_loader,
        loss_func,
        tb_logger=None,
        epochs=10,
        name="tactnet",
        save_dir="data/models",
        use_wandb=False,
        early_stopping_patience=7,
    ):
        """
        Trains the model and validates after each epoch.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            loss_func: Loss function.
            tb_logger: TensorBoard logger.
            epochs (int): Number of training epochs.
            name (str): Name for logging and saving models.
            save_dir (str): Directory to save model checkpoints.
            use_wandb (bool): Whether to use wandb for logging.
            early_stopping_patience (int): Number of epochs with no improvement to wait before stopping.

        Returns:
            TactNet2: The trained model.
        """
        os.makedirs(save_dir, exist_ok=True)

        logger.info(f"Begin TactNet2 training with {epochs} epochs")
        logger.info(f"Training on {self.device}")
        logger.info(f"Using loss function: {loss_func.__class__.__name__}")
        logger.info(f"Using optimizer: {self.optimizer.__class__.__name__}")
        logger.info("Using scheduler: StepLR with step_size=20, gamma=0.1")
        logger.info(f"Saving model checkpoints to {save_dir}")

        # W&B debugging info
        if use_wandb:
            print(f"W&B run initialized: {wandb.run is not None}")
            print(f"W&B run name: {wandb.run.name if wandb.run else 'None'}")
            print(f"W&B project: {wandb.run.project if wandb.run else 'None'}")

            # Test logging
            wandb.log({"test_metric": 1.0})
            print("Test metric logged successfully")

            log_model_summary(self)

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_model_state = None
        epochs_no_improve = 0

        # Lists to track validation metrics across epochs
        val_losses = []
        val_accs = []

        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            train_loop = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                ncols=150,
                desc=f"Training Epoch [{epoch+1}/{epochs}]",
            )

            for batch_idx, (inputs, targets) in train_loop:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = inputs.unsqueeze(1)  # reshape for CNN

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self(inputs)
                loss = loss_func(outputs, targets)

                # Backward pass and optimize
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                self.optimizer.step()

                # Update training statistics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

                current_train_loss = train_loss / (batch_idx + 1)
                current_train_acc = 100.0 * train_correct / train_total

                train_loop.set_postfix(
                    {
                        "loss": f"{current_train_loss:.4f}",
                        "acc": f"{current_train_acc:.2f}%",
                    }
                )

                # Log training metrics to wandb
                if use_wandb and batch_idx % 50 == 0:  # Reduced frequency
                    try:
                        wandb.log(
                            {
                                "train/batch_loss": loss.item(),
                                "train/batch_accuracy": 100.0
                                * predicted.eq(targets).sum().item()
                                / targets.size(0),
                                "train/cumulative_loss": current_train_loss,
                                "train/cumulative_accuracy": current_train_acc,
                                "train/learning_rate": self.optimizer.param_groups[0][
                                    "lr"
                                ],
                                "global_step": epoch * len(train_loader) + batch_idx,
                            }
                        )
                    except Exception as e:
                        print(f"W&B logging error in training: {e}")

            # Step scheduler after each epoch
            self.scheduler.step()

            # Validation phase
            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_val_preds = []
            all_val_targets = []

            val_loop = tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                ncols=150,
                desc=f"Validation Epoch [{epoch+1}/{epochs}]",
            )

            with torch.no_grad():
                for batch_idx, (inputs, targets) in val_loop:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    inputs = inputs.unsqueeze(1)

                    outputs = self(inputs)
                    loss = loss_func(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

                    all_val_preds.extend(predicted.cpu().numpy())
                    all_val_targets.extend(targets.cpu().numpy())

                    current_val_loss = val_loss / (batch_idx + 1)
                    current_val_acc = 100.0 * val_correct / val_total

                    val_loop.set_postfix(
                        {
                            "loss": f"{current_val_loss:.4f}",
                            "acc": f"{current_val_acc:.2f}%",
                        }
                    )

                    # Log validation metrics during validation loop
                    if use_wandb and batch_idx % 20 == 0:  # Reduced frequency
                        try:
                            wandb.log(
                                {
                                    "val/batch_loss": loss.item(),
                                    "val/batch_accuracy": 100.0
                                    * predicted.eq(targets).sum().item()
                                    / targets.size(0),
                                    "val/cumulative_loss": current_val_loss,
                                    "val/cumulative_accuracy": current_val_acc,
                                    "val_global_step": epoch * len(val_loader)
                                    + batch_idx,
                                }
                            )
                        except Exception as e:
                            print(f"W&B logging error in validation: {e}")

                    sleep(0.01)

            # Calculate epoch-level metrics
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = 100.0 * train_correct / train_total
            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc = 100.0 * val_correct / val_total

            # Store validation metrics for tracking stability
            val_losses.append(epoch_val_loss)
            val_accs.append(epoch_val_acc)

            # Log epoch-level metrics to W&B
            if use_wandb:
                try:
                    epoch_metrics = {
                        "epoch": epoch + 1,
                        "epoch/train_loss": epoch_train_loss,
                        "epoch/train_accuracy": epoch_train_acc,
                        "epoch/val_loss": epoch_val_loss,
                        "epoch/val_accuracy": epoch_val_acc,
                        "epoch/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "combined_loss/train": epoch_train_loss,  # For combined loss plot
                        "combined_loss/val": epoch_val_loss,  # For combined loss plot
                        "combined_accuracy/train": epoch_train_acc,  # For combined accuracy plot
                        "combined_accuracy/val": epoch_val_acc,  # For combined accuracy plot
                    }

                    # Calculate stability metrics if enough epochs have passed
                    if epoch >= 3:
                        last_3_val_acc_std = np.std(val_accs[-3:])
                        epoch_metrics[
                            "stability/val_acc_std_last3"
                        ] = last_3_val_acc_std
                        print(
                            f"Validation accuracy stability (std of last 3 epochs): {last_3_val_acc_std:.4f}"
                        )

                    # Log all epoch metrics at once
                    wandb.log(epoch_metrics)
                    print(f"✓ Logged epoch {epoch+1} metrics to W&B")

                    # Log confusion matrix
                    try:
                        log_confusion_matrix(
                            y_true=np.array(all_val_targets),
                            y_pred=np.array(all_val_preds),
                            title=f"Validation Confusion Matrix - Epoch {epoch+1}",
                        )
                    except Exception as e:
                        print(f"Failed to log confusion matrix: {e}")

                except Exception as e:
                    print(f"W&B epoch logging error: {e}")

            # Save model checkpoint
            checkpoint_path = os.path.join(save_dir, f"{name}_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "train_loss": epoch_train_loss,
                    "val_loss": epoch_val_loss,
                    "val_acc": epoch_val_acc,
                },
                checkpoint_path,
            )

            # if use_wandb:
            #     try:
            #         wandb.save(checkpoint_path)
            #     except Exception as e:
            #         print(f"Failed to save checkpoint to W&B: {e}")

            # Check if this is the best model so far
            is_best = False
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                best_val_loss = epoch_val_loss  # Update best_val_loss too
                best_model_state = self.state_dict().copy()
                is_best = True
                epochs_no_improve = 0
            elif epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_state = self.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if is_best:
                best_path = os.path.join(save_dir, f"{name}_best.pth")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "val_loss": epoch_val_loss,
                        "val_acc": epoch_val_acc,
                    },
                    best_path,
                )
                if use_wandb:
                    try:
                        wandb.save(best_path)
                    except Exception as e:
                        print(f"Failed to save best model to W&B: {e}")

                print(
                    f"✓ Saved best model: val_loss={epoch_val_loss:.4f}, val_acc={epoch_val_acc:.2f}%"
                )

            # Print epoch summary
            print(
                f"Epoch {epoch+1}/{epochs} - Train loss: {epoch_train_loss:.4f}, "
                f"Train acc: {epoch_train_acc:.2f}%, "
                f"Val loss: {epoch_val_loss:.4f}, Val acc: {epoch_val_acc:.2f}%, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )

            # Early stopping check
            if epochs_no_improve >= early_stopping_patience:
                print(
                    f"Early stopping triggered after {early_stopping_patience} epochs without improvement"
                )
                break

            # Force W&B sync after each epoch
            if use_wandb:
                try:
                    wandb.run.summary.update(
                        {
                            "current_train_loss": epoch_train_loss,
                            "current_val_loss": epoch_val_loss,
                            "current_train_acc": epoch_train_acc,
                            "current_val_acc": epoch_val_acc,
                            "epochs_completed": epoch + 1,
                        }
                    )
                except Exception as e:
                    print(f"Failed to update W&B summary: {e}")

        # Load best model state at the end of training
        if best_model_state:
            logger.info("Loading best model state")
            self.load_state_dict(best_model_state)

        # Final W&B logging
        if use_wandb:
            try:
                wandb.run.summary.update(
                    {
                        "final_best_val_loss": best_val_loss,
                        "final_best_val_acc": best_val_acc,
                        "total_epochs_trained": epoch + 1,
                        "training_completed": True,
                        "early_stopped": epochs_no_improve >= early_stopping_patience,
                    }
                )
                print("✓ Final W&B summary updated")
            except Exception as e:
                print(f"Failed to finalize W&B logging: {e}")

        return self

    def predict(self, inputs):
        """
        Predicts class labels for input tensor.

        Args:
            inputs (Tensor): Input tensor of shape (B, C, H, W) or (B, H, W).

        Returns:
            Tensor: Predicted class labels of shape (B,).
        """
        self.eval()
        with torch.no_grad():
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)  # Add channel dimension if missing
            inputs = inputs.to(self.device)
            outputs = self(inputs)
            _, predicted = torch.max(outputs, 1)
        return predicted

    def compute_accuracy(self, dataloader):
        """
        Computes the accuracy of the model on a dataset and returns predictions.

        Args:
            dataloader (DataLoader): DataLoader for the dataset.

        Returns:
            tuple:
                - float: Accuracy in the range [0, 1].
                - list[int]: List of predicted class labels.
                - list[int]: List of true labels.
        """
        self.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = inputs.unsqueeze(1)
                preds = self.predict(inputs)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        accuracy = correct / total
        logger.info(f"Accuracy: {accuracy*100:.2f}%")
        return accuracy, all_preds, all_labels
