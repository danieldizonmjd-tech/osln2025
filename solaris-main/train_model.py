from __future__ import annotations

import argparse

from database import init_database
from ml_model import cuda_summary, train_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Starlab solar-array anomaly model.")
    parser.add_argument("--samples", type=int, default=52000)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    init_database()
    print(f"Training device: {cuda_summary()}")
    result = train_and_save(
        samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(f"Saved model: {result.model_path}")
    print(f"Validation accuracy: {result.accuracy:.2%}")
    print(result.report)


if __name__ == "__main__":
    main()
