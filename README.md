---
title: Cell Detection Tool
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
startup_duration_timeout: 5m
suggested_hardware: cpu-basic
pinned: false
license: mit
---

# Visual Cortex - Cell Detection Tool

A Streamlit web application for automated detection and counting of circular cells in microscopy images.

## Features

- **Multi-channel TIFF support**: Load and preview 4-channel microscopy images (up to 1GB)
- **Interactive parameter tuning**: Real-time adjustment of detection parameters
- **Slice preview**: Test settings on small image regions for fast iteration
- **Advanced detection pipeline**: Uses thresholding, morphological operations, and watershed segmentation
- **Export results**: Download annotated images and detection data as CSV

## Usage

1. Upload a .tif/.tiff microscopy image
2. Preview different channels to select the best one for analysis
3. Adjust detection parameters using the settings panel
4. Test on a small slice first for quick feedback
5. Run full detection when satisfied with parameters
6. Download results (overlay image + CSV data)

## Detection Parameters

- **Threshold method**: How to separate cells from background (percentile/otsu/sauvola)
- **Cell separation**: Split touching cells using watershed segmentation
- **Filtering**: Remove false positives based on shape and contrast
- **Size constraints**: Set minimum cell diameter in microns

## Local Development

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
