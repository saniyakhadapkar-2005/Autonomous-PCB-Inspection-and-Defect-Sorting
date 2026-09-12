from pathlib import Path
from collections import Counter

BASE = Path("data/pcb_dataset")

CLASS_NAMES = {
    0: "Damaged Board",
    1: "Damaged Component",
    2: "Missing Component",
    3: "Solder Bridge",
    4: "Solder Ball",
}

splits = ["train", "valid", "test"]

class_counts = Counter()

invalid_labels = []
empty_labels = []
orphan_labels = []

print("=" * 70)
print("PCB DATASET - LABEL VALIDATION")
print("=" * 70)

for split in splits:

    image_dir = BASE / split / "images"
    label_dir = BASE / split / "labels"

    images = list(image_dir.glob("*"))
    image_stems = {img.stem for img in images}

    labels = list(label_dir.glob("*.txt"))
    label_stems = {label.stem for label in labels}

    print(f"\n{'=' * 20} {split.upper()} {'=' * 20}")

    # ---------------------------------------------------------
    # ORPHAN LABEL CHECK
    # ---------------------------------------------------------

    orphan = label_stems - image_stems

    for filename in orphan:
        orphan_labels.append((split, filename))

    print(f"Images : {len(images)}")
    print(f"Labels : {len(labels)}")
    print(f"Orphan Labels : {len(orphan)}")

    # ---------------------------------------------------------
    # LABEL CONTENT CHECK
    # ---------------------------------------------------------

    for label_file in labels:

        # Skip orphan label
        if label_file.stem not in image_stems:
            continue

        try:
            content = label_file.read_text().strip()

            # Empty label
            if not content:
                empty_labels.append((split, label_file.name))
                continue

            for line_number, line in enumerate(content.splitlines(), start=1):

                parts = line.split()

                # YOLO format should contain 5 values
                if len(parts) != 5:
                    invalid_labels.append(
                        (
                            split,
                            label_file.name,
                            line_number,
                            "Wrong number of values"
                        )
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])

                except ValueError:
                    invalid_labels.append(
                        (
                            split,
                            label_file.name,
                            line_number,
                            "Non-numeric value"
                        )
                    )
                    continue

                # Check class ID
                if class_id not in CLASS_NAMES:
                    invalid_labels.append(
                        (
                            split,
                            label_file.name,
                            line_number,
                            f"Invalid class ID: {class_id}"
                        )
                    )
                    continue

                class_counts[class_id] += 1

                # Check bounding box values
                if not (
                    0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < w <= 1
                    and 0 < h <= 1
                ):
                    invalid_labels.append(
                        (
                            split,
                            label_file.name,
                            line_number,
                            "Bounding box outside valid range"
                        )
                    )

        except Exception as e:
            invalid_labels.append(
                (
                    split,
                    label_file.name,
                    0,
                    str(e)
                )
            )


# =============================================================
# RESULTS
# =============================================================

print("\n")
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

total_objects = sum(class_counts.values())

for class_id, class_name in CLASS_NAMES.items():

    count = class_counts[class_id]

    percentage = (
        (count / total_objects) * 100
        if total_objects > 0
        else 0
    )

    print(
        f"{class_id} | "
        f"{class_name:<20} | "
        f"{count:>6} objects | "
        f"{percentage:>6.2f}%"
    )

print(f"\nTotal annotated objects: {total_objects}")


# =============================================================
# INVALID LABELS
# =============================================================

print("\n")
print("=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(f"Invalid label entries : {len(invalid_labels)}")
print(f"Empty label files    : {len(empty_labels)}")
print(f"Orphan label files   : {len(orphan_labels)}")


# =============================================================
# DETAILS
# =============================================================

if invalid_labels:

    print("\nINVALID LABEL EXAMPLES:")

    for item in invalid_labels[:20]:
        print(item)


if empty_labels:

    print("\nEMPTY LABEL EXAMPLES:")

    for item in empty_labels[:20]:
        print(item)


if orphan_labels:

    print("\nORPHAN LABEL EXAMPLES:")

    for item in orphan_labels[:20]:
        print(item)


# =============================================================
# FINAL STATUS
# =============================================================

print("\n")
print("=" * 70)

if (
    len(invalid_labels) == 0
    and len(empty_labels) == 0
    and len(orphan_labels) == 0
):

    print("✓ DATASET VALIDATION PASSED")

else:

    print("⚠ DATASET NEEDS ATTENTION")

print("=" * 70)