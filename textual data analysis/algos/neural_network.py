import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import scipy.sparse as sp
from algos.metrics import evaluate_metrics
from sklearn.preprocessing import LabelEncoder


class NNClassifierModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int, dropout: float = 0.2):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = h
        layers.append(nn.Linear(last_dim, num_classes))  # 最終分類層（不加 activation）
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def sparse_to_dense_tensor(sparse_matrix: sp.csr_matrix) -> torch.Tensor:
    dense = sparse_matrix.toarray()
    return torch.tensor(dense, dtype=torch.float32)

def train_model(
    model: nn.Module,
    x_train: torch.Tensor | sp.csr_matrix | np.ndarray,
    y_train: torch.Tensor | np.ndarray,
    x_val: torch.Tensor | sp.csr_matrix | np.ndarray,
    y_val: torch.Tensor | np.ndarray,
    num_epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    save_path: str = "best_model.pt",
    labels: list | None = None
):
    # Label encoding（共用同一個 encoder）
    if not np.issubdtype(np.array(y_train).dtype, np.integer):
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_val = le.transform(y_val)

    # 轉成 tensor
    if sp.issparse(x_train):
        x_train = sparse_to_dense_tensor(x_train)
    elif not isinstance(x_train, torch.Tensor):
        x_train = torch.tensor(x_train, dtype=torch.float32)

    if sp.issparse(x_val):
        x_val = sparse_to_dense_tensor(x_val)
    elif not isinstance(x_val, torch.Tensor):
        x_val = torch.tensor(x_val, dtype=torch.float32)

    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.long)
    if not isinstance(y_val, torch.Tensor):
        y_val = torch.tensor(y_val, dtype=torch.long)

    # 維度檢查
    assert x_train.shape[0] == y_train.shape[0], "x_train 與 y_train 筆數不一致"
    assert x_val.shape[0] == y_val.shape[0], "x_val 與 y_val 筆數不一致"

    # 訓練設定
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size)

    best_f1 = -1.0
    metrics_per_epoch = []

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim = 1) == yb).sum().item()
            total += xb.size(0)

        acc = correct / total
        avg_loss = total_loss / total

        # 驗證
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(yb)
        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_labels)
        f1_macro = f1_score(y_true, y_pred, average="macro")

        y_pred_labels = le.inverse_transform(y_pred.numpy())
        y_true_labels = le.inverse_transform(y_true.numpy())

        metrics_per_epoch.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "accuracy": acc,
            "f1_macro": f1_macro
        })

        print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1_macro: {f1_macro:.4f}")
        print("Confusion Matrix:\n", evaluate_metrics(y_true_labels, y_pred_labels, labels = labels))
        print(classification_report(y_true_labels, y_pred_labels, digits = 4))

        if f1_macro > best_f1:
            best_f1 = f1_macro
            torch.save(model.state_dict(), save_path)

    best_epoch = max(metrics_per_epoch, key = lambda x: x["f1_macro"])
    print(f"\n最佳模型在 Epoch {best_epoch['epoch']}，F1_macro = {best_epoch['f1_macro']:.4f}")
