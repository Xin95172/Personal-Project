import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
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
    """
    NOTE: Converting the entire sparse matrix to dense can OOM on large vocab.
    Prefer per-row conversion via SparseCSRDataset below.
    """
    dense = sparse_matrix.toarray()
    return torch.tensor(dense, dtype=torch.float32)

class SparseCSRDataset(Dataset):
    """Wrap a CSR matrix and labels, converting rows to dense on-the-fly.

    If labels are non-numeric (e.g., strings), a dummy 0 label is returned to
    keep DataLoader collate happy during inference.
    """
    def __init__(self, X: sp.csr_matrix, y: np.ndarray | None, dtype: torch.dtype = torch.float32):
        assert sp.issparse(X), "X must be a scipy sparse matrix"
        self.X = X.tocsr()
        self.y = None if y is None else np.asarray(y)
        self.dtype = dtype

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        row = self.X[idx]
        x = torch.tensor(row.toarray().ravel(), dtype=self.dtype)
        # Return label if provided; coerce to numeric tensor. If non-numeric, use dummy 0.
        if self.y is None:
            y_t = torch.tensor(0, dtype=torch.long)
        else:
            val = self.y[idx]
            try:
                # np.integer or python int -> make long tensor
                if isinstance(val, (np.integer, int)) or (hasattr(self.y, 'dtype') and self.y.dtype.kind in 'iu'):
                    y_t = torch.tensor(val, dtype=torch.long)
                else:
                    # non-numeric label (e.g., str), fallback to 0
                    y_t = torch.tensor(0, dtype=torch.long)
            except Exception:
                y_t = torch.tensor(0, dtype=torch.long)
        return x, y_t

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
    if False and sp.issparse(x_train):
        x_train = sparse_to_dense_tensor(x_train)
    elif (not isinstance(x_train, torch.Tensor)) and (not sp.issparse(x_train)):
        x_train = torch.tensor(x_train, dtype=torch.float32)

    if False and sp.issparse(x_val):
        x_val = sparse_to_dense_tensor(x_val)
    elif (not isinstance(x_val, torch.Tensor)) and (not sp.issparse(x_val)):
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

    # DataLoaders: support sparse X by densifying per-row on the fly
    def make_loader(X, y, batch_size: int, shuffle: bool):
        if sp.issparse(X):
            return DataLoader(SparseCSRDataset(X, y), batch_size=batch_size, shuffle=shuffle)
        else:
            X_t = X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
            y_t = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)
            return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, batch_size=batch_size, shuffle=False)

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

        # 轉回原始標籤（若使用了 LabelEncoder）
        if 'le' in locals():
            y_pred_labels = le.inverse_transform(y_pred.numpy())
            y_true_labels = le.inverse_transform(y_true.numpy())
        else:
            y_pred_labels = y_pred.numpy()
            y_true_labels = y_true.numpy()

        metrics_per_epoch.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "accuracy": acc,
            "f1_macro": f1_macro
        })

        print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1_macro: {f1_macro:.4f}")
        # 輸出報表與混淆矩陣（避免 UndefinedMetricWarning）
        evaluate_metrics(y_true_labels, y_pred_labels, labels = labels, zero_division = 0)

        if f1_macro > best_f1:
            best_f1 = f1_macro
            torch.save(model.state_dict(), save_path)

    best_epoch = max(metrics_per_epoch, key = lambda x: x["f1_macro"])
    print(f"\n最佳模型在 Epoch {best_epoch['epoch']}，F1_macro = {best_epoch['f1_macro']:.4f}")
