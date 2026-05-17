"""dataset_sanity_check"""

from pathlib import Path
import ast

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from data_utils.iu_xray_dataset import IUXrayDataset


DATA_DIR = Path(r"C:\multimodal_cxt\data\iu_xray")
METADATA_PATH = DATA_DIR / "iu_xray_labeled_metadata.csv"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df = pd.read_csv(METADATA_PATH)
df["labels"] = df["labels"].apply(ast.literal_eval)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_dataset = IUXrayDataset(
    dataframe=train_df,
    tokenizer=tokenizer,
    image_transform=image_transform,
    max_length=128,
)

val_dataset = IUXrayDataset(
    dataframe=val_df,
    tokenizer=tokenizer,
    image_transform=image_transform,
    max_length=128,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

label_cols = [
    "label_effusion",
    "label_cardiomegaly",
    "label_edema",
    "label_pneumonia",
    "label_pneumothorax",
]

print("\nLabel counts:")
print(df[label_cols].sum())

print("\nPositive ratio:")
print(df[label_cols].mean())