# GPQA

**Green Point Cloud Quality Assessment**

A tool for automated point cloud quality assessment that evaluates PLY files and returns objective quality scores.

## Overview

GPQA analyzes point cloud data and provides a Mean Opinion Score (MOS) that quantifies the visual quality of your point cloud. The model outputs a score ranging from 1 to 5, where:

> **Note**: A complete version with additional features and improvements will be released soon.

- **5** - Excellent quality
- **4** - Good quality  
- **3** - Fair quality
- **2** - Poor quality
- **1** - Very poor quality

## Usage

Simply input a PLY file to receive an automated quality assessment score predicted by the model.

## Input Format

- **File type**: `.ply` (Polygon File Format)
- **Output**: MOS score (1.0 - 5.0)

## About the MOS Score

The Mean Opinion Score is a numerical measure of quality derived from the model's evaluation of various point cloud characteristics. Higher scores indicate better overall quality.

---

*For more information, issues, or contributions, please visit the project repository.*