from pathlib import Path

BASE = Path("data/pcb_dataset")

splits = ["train", "valid", "test"]

print("=" * 60)
print("PCB DATASET VERIFICATION")
print("=" * 60)

total_images = 0
total_labels = 0

for split in splits:
    image_dir = BASE / split / "images"
    label_dir = BASE / split / "labels"

    images = list(image_dir.glob("*"))
    labels = list(label_dir.glob("*.txt"))

    print(f"\n{split.upper()}")
    print("-" * 40)
    print(f"Images : {len(images)}")
    print(f"Labels : {len(labels)}")

    total_images += len(images)
    total_labels += len(labels)

print("\n" + "=" * 60)
print(f"TOTAL IMAGES : {total_images}")
print(f"TOTAL LABELS : {total_labels}")
print("=" * 60)

print("\nExpected classes:")
print("0 = Damaged Board")
print("1 = Damaged Component")
print("2 = Missing Component")
print("3 = Solder Bridge")
print("4 = Solder Ball")