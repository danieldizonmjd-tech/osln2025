from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from simulator import ANOMALY_CLASSES, FEATURE_NAMES, synthesize_dataset


MODEL_PATH = Path("models/starlab_anomaly_model.pt")
TRAINING_PROFILE = "starlab_mesh_v5"


class AnomalyNet(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@dataclass
class TrainingResult:
    accuracy: float
    device: str
    model_path: Path
    report: str


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _scaler_payload(scaler: StandardScaler) -> dict:
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "n_features_in": int(scaler.n_features_in_),
    }


def _scaler_from_payload(payload: dict) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.array(payload["mean"], dtype=np.float32)
    scaler.scale_ = np.array(payload["scale"], dtype=np.float32)
    scaler.var_ = np.array(payload["var"], dtype=np.float32)
    scaler.n_features_in_ = int(payload["n_features_in"])
    return scaler


def train_and_save(
    samples: int = 28000,
    epochs: int = 22,
    batch_size: int = 256,
    model_path: Path = MODEL_PATH,
) -> TrainingResult:
    device = get_device()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    rows, labels = synthesize_dataset(samples=samples)
    x = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.18, random_state=11, stratify=y
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)

    train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = AnomalyNet(len(FEATURE_NAMES), len(ANOMALY_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0025, weight_decay=0.0008)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_tensor = torch.from_numpy(x_val).to(device)
        logits = model(val_tensor).cpu()
        predictions = torch.argmax(logits, dim=1).numpy()

    accuracy = float(accuracy_score(y_val, predictions))
    report = classification_report(
        y_val,
        predictions,
        target_names=ANOMALY_CLASSES,
        zero_division=0,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "classes": ANOMALY_CLASSES,
            "scaler": _scaler_payload(scaler),
            "accuracy": accuracy,
            "device": str(device),
            "training_profile": TRAINING_PROFILE,
        },
        model_path,
    )
    return TrainingResult(accuracy=accuracy, device=str(device), model_path=model_path, report=report)


class StarlabAnomalyModel:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self.device = get_device()
        self.model = AnomalyNet(len(FEATURE_NAMES), len(ANOMALY_CLASSES)).to(self.device)
        self.scaler: StandardScaler | None = None
        self.accuracy = 0.0
        self.loaded = False

    @property
    def cuda_enabled(self) -> bool:
        return self.device.type == "cuda"

    def load(self) -> None:
        if not self.model_path.exists():
            train_and_save(model_path=self.model_path, samples=52000, epochs=28)
        payload = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if payload.get("training_profile") != TRAINING_PROFILE:
            train_and_save(model_path=self.model_path, samples=52000, epochs=28)
            payload = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["state_dict"])
        self.scaler = _scaler_from_payload(payload["scaler"])
        self.accuracy = float(payload.get("accuracy", 0.0))
        self.model.eval()
        self.loaded = True

    def predict(self, features: list[float]) -> tuple[str, float, dict[str, float]]:
        if not self.loaded:
            self.load()
        assert self.scaler is not None

        x = np.asarray([features], dtype=np.float32)
        x = self.scaler.transform(x).astype(np.float32)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x).to(self.device))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

        index = int(np.argmax(probabilities))
        scores = {
            label: float(probabilities[i])
            for i, label in enumerate(ANOMALY_CLASSES)
        }
        return ANOMALY_CLASSES[index], float(probabilities[index]), scores


def cuda_summary() -> str:
    if not torch.cuda.is_available():
        return "CPU fallback"
    return f"CUDA: {torch.cuda.get_device_name(0)}"
