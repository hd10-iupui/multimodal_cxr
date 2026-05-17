from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from data_utils.iu_xray_dataset import IUXrayDataset


DATA_DIR = Path(r"C:\multimodal_cxt\data\iu_xray")
METADATA_PATH = DATA_DIR / "iu_xray_labeled_metadata.csv"

df = pd.read_csv(METADATA_PATH)

# in case labels become string after read from csv, transfer it to list
import ast
df["labels"] = df["labels"].apply(ast.literal_eval)

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = IUXrayDataset(
    dataframe=df,
    tokenizer=tokenizer,
    image_transform=image_transform,
    max_length=128
)

# 1. test single sample
sample = dataset[0]

print("Single sample:")
print("pixel_values:", sample["pixel_values"].shape)
print("input_ids:", sample["input_ids"].shape)
print("attention_mask:", sample["attention_mask"].shape)
print("labels:", sample["labels"])

# 2. test batch
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

batch = next(iter(dataloader))

print("\nBatch:")
print("pixel_values:", batch["pixel_values"].shape)
print("input_ids:", batch["input_ids"].shape)
print("attention_mask:", batch["attention_mask"].shape)
print("labels:", batch["labels"].shape)