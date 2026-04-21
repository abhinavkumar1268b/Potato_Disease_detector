# Dataset Folder Structure

This project uses a 4-class image dataset for potato leaf disease classification.

## Required Folder Layout

```
Potato disease/
│
├── Data set/
│   ├── Potato___Early_blight/     # ~1000 images of early blight leaves
│   ├── Potato___Late_blight/      # ~1000 images of late blight leaves
│   ├── Potato___healthy/          # ~152  images of healthy leaves
│   └── Not_Potato_Leaf/           # ~250+ images of non-leaf objects
│
├── app/
│   ├── app.py                     # Streamlit app
│   ├── potato_disease_model_v2.keras  # Trained model (copied here after training)
│   └── requirements.txt           # Python dependencies
│
├── train.py                       # Training script (run this to train)
├── download_not_leaf_images.py    # Helper: download Not_Potato_Leaf images
├── potato_disease_model_v2.keras  # Trained model output
├── training_results.png           # Accuracy/loss plot (generated after training)
└── DATASET_STRUCTURE.md           # This file
```

## Class Names (as detected by TensorFlow)

TensorFlow's `image_dataset_from_directory` reads class names from subfolder names,
sorted alphabetically:

| Index | Folder Name              | Display Name     |
|-------|--------------------------|------------------|
| 0     | `Not_Potato_Leaf`        | Not Potato Leaf  |
| 1     | `Potato___Early_blight`  | Early Blight     |
| 2     | `Potato___Late_blight`   | Late Blight      |
| 3     | `Potato___healthy`       | Healthy          |

> **Note:** The order is alphabetical. The `app.py` uses
> `CLASS_NAMES = dataset.class_names` automatically, so no manual mapping is needed.

## Sourcing Not_Potato_Leaf Images

Run the helper script to automatically download ~250 diverse images:

```bash
python download_not_leaf_images.py
```

For best model accuracy, aim for **500–1000** non-leaf images covering diverse
categories: animals, vehicles, food, scenery, household items, etc.

## Image Requirements

- Format: JPG, JPEG, or PNG
- Minimum size: 50×50 pixels (will be resized to 256×256 during training)
- No corrupted files

## Steps to Train

1. **Populate the dataset** (run once):
   ```bash
   python download_not_leaf_images.py
   ```

2. **Train the model**:
   ```bash
   python train.py
   ```

3. **Run the Streamlit app**:
   ```bash
   cd app
   streamlit run app.py
   ```
