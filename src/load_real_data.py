import os
import json
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset

print("Loading real Cantonese dataset from mirror...")

dataset = load_dataset(
    'leeduckgo/cantonese-life-scenarios-corpus',
    split='train',
    streaming=True
)

print("Dataset loaded successfully!")

samples = []
for i, sample in enumerate(dataset):
    if i >= 50:
        break
    text = sample.get('text', '')
    if text and len(text) > 2:
        samples.append({'text': text})
        print(f"  [{i+1}] {text[:30]}")

os.makedirs('data', exist_ok=True)
with open('data/real_texts.json', 'w', encoding='utf-8') as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(samples)} samples to data/real_texts.json")
