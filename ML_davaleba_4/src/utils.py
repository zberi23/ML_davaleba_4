"""
ML Davaleba 4 — utility functions
data loading, training loop, WandB logging, evaluation, plotting
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import wandb


EMOTION_LABELS = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
    4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}
NUM_CLASSES = 7

# FER2013 statistics (computed in notebook 01)
FER_MEAN = 0.5077
FER_STD = 0.2550


class FERDataset(Dataset):
    """
    FER2013 dataset wrapper.
    X: (N, 48, 48) uint8 arrays
    y: (N,) int64 emotion labels (0-6)
    """
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]  # (48, 48) uint8
        label = int(self.y[idx])

        if self.transform is not None:
            img = self.transform(img)
        else:
            # default: (1, 48, 48) float in [0, 1]
            img = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)

        return img, label


def load_fer2013(data_path):
    """Loads preprocessed FER2013 .npz file."""
    data = np.load(data_path)
    return (data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            data['X_test'], data['y_test'])


def get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                    batch_size=64, train_transform=None, eval_transform=None,
                    num_workers=2):
    """Creates train/val/test DataLoaders."""
    train_ds = FERDataset(X_train, y_train, transform=train_transform)
    val_ds = FERDataset(X_val, y_val, transform=eval_transform)
    test_ds = FERDataset(X_test, y_test, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def count_parameters(model):
    """Counts trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sanity_check_forward(model, loader, criterion, device):
    """
    Forward pass sanity check.
    Untrained model output should give loss ~= log(num_classes) = log(7) ~= 1.9459
    if logits are roughly uniform.
    """
    model.eval()
    X, y = next(iter(loader))
    X, y = X.to(device), y.to(device)

    with torch.no_grad():
        out = model(X)
        loss = criterion(out, y).item()

    expected = float(np.log(NUM_CLASSES))

    print(f"=== Forward Pass Sanity Check ===")
    print(f"  Batch input shape:  {tuple(X.shape)}")
    print(f"  Batch output shape: {tuple(out.shape)}")
    print(f"  Initial loss:       {loss:.4f}")
    print(f"  Expected (random):  {expected:.4f}  = log({NUM_CLASSES})")
    print(f"  Diff:               {abs(loss - expected):.4f}")
    print(f"  Trainable params:   {count_parameters(model):,}")
    return loss


def sanity_check_overfit_batch(model, loader, optimizer, criterion, device,
                                num_steps=100):
    """
    Overfit a single batch — sanity check that model can learn.
    If loss doesn't drop close to 0 here, model/training has a bug.
    """
    model.train()
    X, y = next(iter(loader))
    X, y = X.to(device), y.to(device)

    print(f"=== Overfit Single Batch Sanity Check ({num_steps} steps) ===")
    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % (num_steps // 10) == 0 or step == num_steps - 1:
            acc = (out.argmax(1) == y).float().mean().item()
            print(f"  Step {step:4d} | loss {loss.item():.4f} | acc {acc:.4f}")

    final_acc = (out.argmax(1) == y).float().mean().item()
    print(f"  Final batch accuracy: {final_acc:.4f}")
    if final_acc > 0.95:
        print(f"  Model can overfit a single batch - capacity OK")
    else:
        print(f"  Warning: model failed to overfit batch - check capacity/lr")

    return losses


def get_gradient_norm(model):
    """Computes total L2 norm of gradients across all parameters."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    return total_norm_sq ** 0.5


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Trains one epoch. Returns (avg_loss, accuracy, mean_grad_norm)."""
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    grad_norms = []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()

        grad_norms.append(get_gradient_norm(model))
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
        total_count += X.size(0)

    return (total_loss / total_count,
            total_correct / total_count,
            float(np.mean(grad_norms)))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluates on a loader. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * X.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
        total_count += X.size(0)
    return total_loss / total_count, total_correct / total_count


def train_model(model, train_loader, val_loader, optimizer, criterion, device,
                num_epochs=20, scheduler=None, run_name=None,
                project="ml_davaleba_4", config=None, save_path=None,
                early_stop_patience=None, verbose=True):
    """
    Full training loop with WandB logging.
    Returns: (history dict, best_val_acc)
    """
    if wandb.run is not None:
        wandb.finish()
    wandb.init(project=project, name=run_name, config=config or {}, reinit=True)

    history = {'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': [],
               'grad_norm': [], 'lr': []}

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_acc, grad_norm = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['grad_norm'].append(grad_norm)
        history['lr'].append(current_lr)

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'grad_norm': grad_norm,
            'lr': current_lr,
        })

        if verbose:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val_loss {val_loss:.4f} acc {val_acc:.4f} | "
                  f"grad {grad_norm:.3f} | lr {current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if early_stop_patience is not None and patience_counter >= early_stop_patience:
                if verbose:
                    print(f"Early stop at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)")
                break

    wandb.log({'best_val_acc': best_val_acc})
    wandb.summary['best_val_acc'] = best_val_acc
    return history, best_val_acc


def plot_training_curves(history, title="Training Curves"):
    """Plots train/val loss, accuracy, and gradient norms."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(history['train_loss'], label='train')
    axes[0].plot(history['val_loss'], label='val')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].set_title('Loss'); axes[0].grid(alpha=0.3)

    axes[1].plot(history['train_acc'], label='train')
    axes[1].plot(history['val_acc'], label='val')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].set_title('Accuracy'); axes[1].grid(alpha=0.3)

    axes[2].plot(history['grad_norm'], color='darkred')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Mean Grad Norm')
    axes[2].set_title('Gradient Norms'); axes[2].grid(alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def get_predictions(model, loader, device):
    """Returns (preds, labels) numpy arrays over a loader."""
    model.eval()
    all_preds, all_labels = [], []
    for X, y in loader:
        X = X.to(device)
        out = model(X)
        all_preds.append(out.argmax(1).cpu().numpy())
        all_labels.append(np.asarray(y))
    return np.concatenate(all_preds), np.concatenate(all_labels)


def plot_confusion_matrix(preds, labels, title="Confusion Matrix",
                          log_to_wandb=False, normalize=True):
    """Plots a confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    if normalize:
        cm_plot = cm / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        cm_plot = cm
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=[EMOTION_LABELS[i] for i in range(7)],
                yticklabels=[EMOTION_LABELS[i] for i in range(7)], ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
    plt.tight_layout()

    if log_to_wandb and wandb.run is not None:
        wandb.log({title: wandb.Image(fig)})

    plt.show()
    return cm


def print_classification_report(preds, labels):
    """Prints per-class precision / recall / f1."""
    target_names = [EMOTION_LABELS[i] for i in range(7)]
    print(classification_report(labels, preds, target_names=target_names, digits=4))


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
