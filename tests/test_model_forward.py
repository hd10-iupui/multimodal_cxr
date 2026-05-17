from pathlib import Path
import ast

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from data_utils.iu_xray_dataset import IUXrayDataset
from models.multimodal_model import MultimodalCXRModel


DATA_DIR = Path(r"C:\multimodal_cxt\data\iu_xray")
METADATA_PATH = DATA_DIR / "iu_xray_labeled_metadata.csv"


def main():
    df = pd.read_csv(METADATA_PATH)
    df["labels"] = df["labels"].apply(ast.literal_eval)

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = IUXrayDataset(
        dataframe=df,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_length=128,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(dataloader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = MultimodalCXRModel(num_labels=5).to(device)
    model.eval()

    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    logits = outputs["logits"]
    attention_weights = outputs["attention_weights"]

    print("logits shape:", logits.shape)
    print("attention_weights shape:", attention_weights.shape)

    probs = torch.sigmoid(logits)
    print("probabilities:", probs)


if __name__ == "__main__":
    main()