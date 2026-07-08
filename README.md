age_estimation/
├── data/
│   ├── raw/           # UTKFace original, untouched
│   ├── masked/        # Your masked variants go here
│   └── processed/     # Cleaned, resized, normalized
├── models/
│   └── checkpoints/   # Saved model weights
├── src/
│   ├── dataset.py     # PyTorch Dataset class
│   ├── model.py       # Model architecture
│   ├── train.py       # Training logic
│   ├── evaluate.py    # Evaluation logic
│   └── masking.py     # Sapiens + masking pipeline
├── main.py            # Click CLI entry point
├── requirements.txt   # All your dependencies
└── README.md
(ai generated)