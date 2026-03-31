# Computer Vision for Manufacturing

Computer vision datasets and notebooks for mining, manufacturing, and safety applications on Databricks.

## Datasets

Three diverse datasets with permissive licences have been loaded into the Databricks Unity Catalog volume `brian_gen_ai.cv_manufacturing.raw`:

### 1. SHWD — Safety Helmet Wearing Dataset

- **Use case:** PPE / safety helmet detection
- **Size:** 7,581 images with 9,044 helmet and 111,514 head annotations
- **Format:** Pascal VOC (XML annotations + JPEG images)
- **Licence:** MIT
- **Source:** [njvisionpower/Safety-Helmet-Wearing-Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset) (GitHub, 1,672+ stars)
- **Volume path:** `/Volumes/brian_gen_ai/cv_manufacturing/raw/shwd_safety_helmet/`

### 2. DeepPCB — PCB Defect Detection

- **Use case:** Electronics manufacturing defect inspection
- **Size:** 1,500 image pairs (template + test), 6 defect types (open, short, mousebite, spur, copper, pin-hole)
- **Format:** Custom (test/template image pairs with txt bounding box annotations)
- **Licence:** MIT
- **Source:** [tangsanli5201/DeepPCB](https://github.com/tangsanli5201/DeepPCB) (GitHub, 472+ stars)
- **Volume path:** `/Volumes/brian_gen_ai/cv_manufacturing/raw/deep_pcb_defects/`

### 3. Corrosion Detection

- **Use case:** Infrastructure / asset corrosion detection
- **Size:** ~9,200 images with bounding box annotations
- **Format:** HuggingFace Parquet (embedded images + annotations)
- **Licence:** CC BY 4.0
- **Source:** [Francesco/corrosion-detection](https://huggingface.co/datasets/Francesco/corrosion-detection) (HuggingFace)
- **Volume path:** `/Volumes/brian_gen_ai/cv_manufacturing/raw/corrosion_detection/`

## Databricks Setup

```
Catalog:  brian_gen_ai
Schema:   cv_manufacturing
Volume:   raw (MANAGED)
```

## Additional Dataset Research

A broader survey of 70+ CV datasets for mining/manufacturing/safety (with licence info and OSS sources) is available in the research notes.

## Licence

See [LICENSE](LICENSE) for repository licence.
