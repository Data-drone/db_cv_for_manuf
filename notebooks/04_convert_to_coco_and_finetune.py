# Databricks notebook source
# MAGIC %md
# MAGIC # Convert Datasets to COCO Format & Test Finetuning
# MAGIC
# MAGIC This notebook converts the SHWD (Safety Helmet Wearing Dataset) from Pascal VOC
# MAGIC format to COCO format, stores it in a UC volume, and then tests the SAM 3.1
# MAGIC finetuning pipeline.
# MAGIC
# MAGIC **Input:** `/Volumes/brian_gen_ai/cv_manufacturing/raw/shwd_safety_helmet/shwd_voc2028.zip`
# MAGIC **Output:** `/Volumes/brian_gen_ai/cv_manufacturing/coco_datasets/shwd/`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

# MAGIC %pip install pycocotools lxml
# MAGIC %restart_python

# COMMAND ----------

import os
import json
import shutil
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

# Paths
UC_VOLUME_RAW = "/Volumes/brian_gen_ai/cv_manufacturing/raw"
UC_VOLUME_COCO = "/Volumes/brian_gen_ai/cv_manufacturing/coco_datasets"
WORK_DIR = "/tmp/coco_convert"

os.makedirs(WORK_DIR, exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create the coco_datasets volume (if needed)

# COMMAND ----------

spark.sql("CREATE VOLUME IF NOT EXISTS brian_gen_ai.cv_manufacturing.coco_datasets")
print(f"Volume ready: {UC_VOLUME_COCO}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Extract SHWD VOC dataset

# COMMAND ----------

shwd_zip = os.path.join(UC_VOLUME_RAW, "shwd_safety_helmet", "shwd_voc2028.zip")
shwd_extract = os.path.join(WORK_DIR, "shwd_voc")

if not os.path.exists(os.path.join(shwd_extract, "VOC2028")):
    print(f"Extracting {shwd_zip}...")
    with zipfile.ZipFile(shwd_zip, 'r') as zf:
        zf.extractall(shwd_extract)
    print("Extraction complete")
else:
    print("Already extracted")

# Verify structure
voc_root = os.path.join(shwd_extract, "VOC2028")
for d in ["Annotations", "JPEGImages", "ImageSets"]:
    p = os.path.join(voc_root, d)
    if os.path.exists(p):
        count = len(os.listdir(p))
        print(f"  {d}/: {count} files")
    else:
        print(f"  {d}/: MISSING")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Read train/test splits from VOC ImageSets

# COMMAND ----------

def read_imageset(voc_root, split_name):
    """Read image IDs from a VOC ImageSets/Main split file."""
    # Try different paths
    for subdir in ["Main", "Layout", ""]:
        path = os.path.join(voc_root, "ImageSets", subdir, f"{split_name}.txt") if subdir else os.path.join(voc_root, "ImageSets", f"{split_name}.txt")
        if os.path.exists(path):
            with open(path) as f:
                ids = [line.strip().split()[0] for line in f if line.strip()]
            return ids
    return []

train_ids = read_imageset(voc_root, "train")
val_ids = read_imageset(voc_root, "val")
test_ids = read_imageset(voc_root, "test")
trainval_ids = read_imageset(voc_root, "trainval")

print(f"Train: {len(train_ids)} images")
print(f"Val: {len(val_ids)} images")
print(f"Test: {len(test_ids)} images")
print(f"Trainval: {len(trainval_ids)} images")

# If no explicit train/test split, use trainval for train and val for test
if not train_ids and trainval_ids:
    train_ids = trainval_ids
if not test_ids and val_ids:
    test_ids = val_ids

# If still no split, do an 80/20 split of all available annotations
if not train_ids:
    all_xmls = [f.replace('.xml', '') for f in os.listdir(os.path.join(voc_root, "Annotations")) if f.endswith('.xml')]
    import random
    random.seed(42)
    random.shuffle(all_xmls)
    split_idx = int(len(all_xmls) * 0.8)
    train_ids = all_xmls[:split_idx]
    test_ids = all_xmls[split_idx:]

print(f"\nUsing: Train={len(train_ids)}, Test={len(test_ids)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Convert VOC XML to COCO JSON

# COMMAND ----------

def voc_to_coco(voc_root, image_ids, output_dir, split_name="train"):
    """Convert Pascal VOC annotations to COCO format."""
    ann_dir = os.path.join(voc_root, "Annotations")
    img_dir = os.path.join(voc_root, "JPEGImages")
    out_img_dir = os.path.join(output_dir, split_name)
    os.makedirs(out_img_dir, exist_ok=True)

    # Collect all category names first
    category_names = set()
    for img_id in image_ids:
        xml_path = os.path.join(ann_dir, f"{img_id}.xml")
        if not os.path.exists(xml_path):
            continue
        tree = ET.parse(xml_path)
        for obj in tree.findall(".//object"):
            name = obj.find("name").text.strip()
            category_names.add(name)

    category_names = sorted(category_names)
    cat_to_id = {name: i + 1 for i, name in enumerate(category_names)}

    categories = [
        {"id": cat_to_id[name], "name": name, "supercategory": "object"}
        for name in category_names
    ]

    images = []
    annotations = []
    ann_id = 1
    skipped = 0
    copied = 0

    for idx, img_id in enumerate(image_ids):
        xml_path = os.path.join(ann_dir, f"{img_id}.xml")
        if not os.path.exists(xml_path):
            skipped += 1
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image filename
        fname_elem = root.find("filename")
        if fname_elem is not None:
            filename = fname_elem.text.strip()
        else:
            filename = f"{img_id}.jpg"

        # Get image size
        size = root.find("size")
        if size is not None:
            width = int(size.find("width").text)
            height = int(size.find("height").text)
        else:
            width, height = 0, 0

        # Copy image
        src_img = os.path.join(img_dir, filename)
        if not os.path.exists(src_img):
            # Try common extensions
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG"]:
                alt = os.path.join(img_dir, f"{img_id}{ext}")
                if os.path.exists(alt):
                    src_img = alt
                    filename = f"{img_id}{ext}"
                    break

        if os.path.exists(src_img):
            dst_img = os.path.join(out_img_dir, filename)
            if not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)
            copied += 1
        else:
            skipped += 1
            continue

        image_id = idx + 1
        images.append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
        })

        # Parse objects
        for obj in root.findall(".//object"):
            name = obj.find("name").text.strip()
            cat_id = cat_to_id[name]

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            w = xmax - xmin
            h = ymax - ymin

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    # Save COCO JSON
    ann_file = os.path.join(out_img_dir, "_annotations.coco.json")
    with open(ann_file, "w") as f:
        json.dump(coco, f)

    print(f"  {split_name}: {len(images)} images, {len(annotations)} annotations, "
          f"{len(categories)} categories (skipped {skipped}, copied {copied})")

    return coco

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Run conversion

# COMMAND ----------

# Convert to local temp first
coco_local = os.path.join(WORK_DIR, "shwd_coco")
os.makedirs(coco_local, exist_ok=True)

print("Converting train split...")
train_coco = voc_to_coco(voc_root, train_ids, coco_local, "train")

print("Converting test split...")
test_coco = voc_to_coco(voc_root, test_ids, coco_local, "test")

print(f"\nCategories: {[c['name'] for c in train_coco['categories']]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Copy to UC volume

# COMMAND ----------

# Copy the COCO dataset to the UC volume
uc_shwd = os.path.join(UC_VOLUME_COCO, "shwd")

for split in ["train", "test"]:
    src = os.path.join(coco_local, split)
    dst = os.path.join(uc_shwd, split)

    if os.path.exists(dst):
        print(f"  {split}/ already exists in volume, skipping copy")
        # Verify annotation file
        ann = os.path.join(dst, "_annotations.coco.json")
        if os.path.exists(ann):
            with open(ann) as f:
                c = json.load(f)
            print(f"    -> {len(c['images'])} images, {len(c['annotations'])} annotations")
    else:
        print(f"  Copying {split}/ to volume...")
        shutil.copytree(src, dst)
        print(f"    -> Done")

print(f"\nCOCO dataset ready at: {uc_shwd}")
print(f"  train/: images + _annotations.coco.json")
print(f"  test/:  images + _annotations.coco.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Verify the COCO dataset

# COMMAND ----------

for split in ["train", "test"]:
    ann_path = os.path.join(uc_shwd, split, "_annotations.coco.json")
    with open(ann_path) as f:
        coco = json.load(f)

    print(f"\n{split.upper()} split:")
    print(f"  Images: {len(coco['images'])}")
    print(f"  Annotations: {len(coco['annotations'])}")
    print(f"  Categories: {[c['name'] for c in coco['categories']]}")

    # Category distribution
    cat_counts = defaultdict(int)
    for ann in coco['annotations']:
        cat_id = ann['category_id']
        cat_name = next(c['name'] for c in coco['categories'] if c['id'] == cat_id)
        cat_counts[cat_name] += 1
    for name, count in sorted(cat_counts.items()):
        print(f"    {name}: {count} annotations")

    # Check a few images exist
    img_dir = os.path.join(uc_shwd, split)
    sample = coco['images'][:3]
    for img in sample:
        img_path = os.path.join(img_dir, img['file_name'])
        exists = os.path.exists(img_path)
        size = os.path.getsize(img_path) if exists else 0
        print(f"    Sample: {img['file_name']} ({img['width']}x{img['height']}) -> exists={exists}, {size/1024:.0f}KB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary
# MAGIC
# MAGIC SHWD dataset converted from Pascal VOC to COCO format and stored in:
# MAGIC ```
# MAGIC /Volumes/brian_gen_ai/cv_manufacturing/coco_datasets/shwd/
# MAGIC   train/
# MAGIC     <images>
# MAGIC     _annotations.coco.json
# MAGIC   test/
# MAGIC     <images>
# MAGIC     _annotations.coco.json
# MAGIC ```
# MAGIC
# MAGIC This is ready for use with the SAM 3.1 finetuning notebook (03).
# MAGIC Set `DATASET_ROOT` to `/Volumes/brian_gen_ai/cv_manufacturing/coco_datasets/shwd`
# MAGIC in the finetuning notebook.
