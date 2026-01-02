# Data Folder

This folder contains data files needed by ST-Bot.

## Required Files

### categorical_values.json
Contains product categories, subcategories, and other categorical metadata.

Copy from Work-Bot system:
```bash
cp ../Work-Bot/documents/categorical_values.json ./categorical_values.json
```

### sku_vocab.json
Contains valid product SKUs for extraction.

Copy from Work-Bot system:
```bash
cp ../Work-Bot/documents/sku_vocab.json ./sku_vocab.json
```

## File Formats

### categorical_values.json
```json
{
  "category": ["cables", "docking stations", "..."],
  "subcategory": ["hdmi cables", "usb-c docks", "..."],
  "material": ["aluminum", "plastic", "..."],
  "color": ["black", "white", "..."],
  ...
}
```

### sku_vocab.json
```json
{
  "skus": ["CDP2DPMM6B", "CDP2DPMM1MB", "..."]
}
```
