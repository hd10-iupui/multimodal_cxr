import pandas as pd
from pathlib import Path
from PIL import Image

DATA_DIR = Path(r'C:\multimodal_cxt\data\iu_xray')
IMAGE_DIR = DATA_DIR / 'images'/'images_normalized'

reports_df = pd.read_csv(DATA_DIR / 'indiana_reports.csv')
projections_df = pd.read_csv(DATA_DIR / 'indiana_projections.csv')

print(reports_df.head())
print(projections_df.head())

# multimodal paired dataset
merged_df = reports_df.merge(projections_df, on = 'uid')

# look at the types of projection
print(merged_df['projection'].value_counts())

# use only frontal for phase 1
merged_df = merged_df[merged_df['projection'].str.lower().str.contains('frontal', na=False)].copy()

# construct image path / multimodal alignment
# merged_df['image_path'] = ('C:\multimodal_cxt\data\iu_xray\images\images_normalized/' + merged_df['filename'])
merged_df['image_path'] = merged_df['filename'].apply(lambda x: str(IMAGE_DIR / x))

# use findings as text; impression is used as label later
merged_df['text'] = (merged_df['findings'].fillna('')).str.strip()

# filter the samples without text
merged_df = merged_df[merged_df['text'] != ''].copy()

print(merged_df.head())
print(len(merged_df))
print(merged_df.iloc[0]['findings'])
print(merged_df.iloc[0]['text'])
print(merged_df.iloc[0]['image_path'])

img = Image.open(merged_df.iloc[0]['image_path']).convert('RGB')
print(img.size)

# use 5 freq abnormality as label in phase 1
# LABELS = ['effusion', 'cardiomegaly', 'edema', 'pneumonia', 'pneumothorax']

LABEL_RULES = {
    "effusion": {
        "positive": ["effusion", "pleural fluid"],
        "negative": ["no pleural effusion", "without pleural effusion", "no effusion"]
    },
    "cardiomegaly": {
        "positive": ["cardiomegaly", "enlarged cardiac silhouette", "cardiac enlargement"],
        "negative": ["no cardiomegaly", "heart size is normal", "normal heart size"]
    },
    "edema": {
        "positive": ["edema", "pulmonary edema", "vascular congestion"],
        "negative": ["no pulmonary edema", "no edema"]
    },
    "pneumonia": {
        "positive": ["pneumonia", "infectious infiltrate"],
        "negative": ["no pneumonia"]
    },
    "pneumothorax": {
        "positive": ["pneumothorax"],
        "negative": ["no pneumothorax", "without pneumothorax"]
    }
}

import re

NEGATION_PATTERNS = [
    r"\bno\b(?:\W+\w+){{0,6}}\W+{term}",
    r"\bwithout\b(?:\W+\w+){{0,6}}\W+{term}",
    r"\bnegative for\b(?:\W+\w+){{0,6}}\W+{term}",
    r"\bno evidence of\b(?:\W+\w+){{0,6}}\W+{term}",
]


def has_negated_term(text, term):
    text = str(text).lower()
    escaped_term = re.escape(term)

    for pattern in NEGATION_PATTERNS:
        regex = pattern.format(term=escaped_term)
        if re.search(regex, text):
            return True

    return False


# generate label
def extract_label(text, label_name):
    text = str(text).lower()
    rules = LABEL_RULES[label_name]

    # negative priority
    for pattern in rules['negative']:
        if pattern in text:
            return 0

    # regex negation
    for term in rules["positive"]:
        if has_negated_term(text, term):
            return 0

    # positive
    for pattern in rules['positive']:
        if pattern in text:
            return 1

    return 0


for label_name in LABEL_RULES.keys():
    merged_df[f'label_{label_name}'] = (merged_df['impression'].apply(lambda x: extract_label(x, label_name)))

LABEL_COLS = [f'label_{label}' for label in LABEL_RULES.keys()]

merged_df['labels'] = (merged_df[LABEL_COLS].values.tolist())

print(merged_df.iloc[0]['labels'])  # [0, 0, 0, 0, 0]

# save the metadata
output_path = DATA_DIR / "iu_xray_labeled_metadata.csv"
merged_df.to_csv(output_path, index=False)
print(f"Saved to {output_path}")