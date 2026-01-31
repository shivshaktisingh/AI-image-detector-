from datasets import load_dataset
import os
from PIL import Image
import tqdm

# Hugging Face se dataset load karo
print("🔄 Loading dataset...")
dataset = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train")

# Output folder
output_dir = "dataset"
ai_dir = os.path.join(output_dir, "ai")
real_dir = os.path.join(output_dir, "real")
os.makedirs(ai_dir, exist_ok=True)
os.makedirs(real_dir, exist_ok=True)

# Label mapping (0 = AI, 1 = Real)
label_map = {0: "ai", 1: "real"}

print("📥 Saving images to folders...")
for i, row in enumerate(tqdm.tqdm(dataset)):
    img = row["image"]   # Ye already PIL.Image hai
    label = row["label"]

    try:
        # Save into respective folder
        filename = f"{i}.jpg"
        folder = ai_dir if label == 0 else real_dir
        img.save(os.path.join(folder, filename))
    except Exception as e:
        print(f"⚠️ Error saving image {i}: {e}")

print("✅ Dataset download complete!")
print(f"AI images saved in: {ai_dir}")
print(f"Real images saved in: {real_dir}")
