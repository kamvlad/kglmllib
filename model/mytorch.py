import numpy as np
import torch

# Warning metric(gt, pred)

def train_epoch(train_loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for X, y in train_loader:
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
    return running_loss / len(train_loader.dataset)


def train_loop(
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        metrics,  # Warning. Metric calculation is metic(gt, pred)
        metric_direction,
        last_epoch,
        

        calculate_metric_on_train,

        ):
    pass


def validate_metrics(val_loader, model, metrics):
    running_metrics = {}
    model.eval()
    with torch.no_grad():
        for X, y in val_loader:
            outputs = model(X)
            for name, metric in metrics:
                running_metrics[name] = running_metrics.get(name, 0.0) + metric(y, outputs) * X.size(0)

    return {k: v / len(val_loader.dataset) for k, v in running_metrics.items()}

class ColumnarDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, device='cuda'):
        self.X = torch.Tensor(X).to(device)
        if len(y.shape) == 1:
            y = np.expand_dims(y, 1)
        self.y = torch.Tensor(y).to(device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]