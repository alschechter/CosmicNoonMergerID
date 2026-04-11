
# CosmicNoonMergerID
<img width="1709" height="1438" alt="paper_figure_categories" src="https://github.com/user-attachments/assets/31ae9887-cdeb-4a9e-b75c-08e4d5dfa702" />

## Overview
This work uses TNG50, HST CANDELS imaging, and Zoobot to create a CNN to identify galaxy mergers near cosmic noon. This was accepted to NeurIPS ML4PS 2025.

### Data Availability
All datasets used in this project are available on our [Zenodo Page](https://zenodo.org/records/17612012).

---

## Installation

To set up the environment, install the required dependencies using:

```bash
pip install -r requirements.txt
```
or consult appropriate online documentation.

## Code Structure

The repository is organized into the following components:
- **Data Preparation**:
  - `TNGstuff`
    Code written by Dr. Rebecca Nevin to walk the TNG merger trees and select galaxy mergers and mass matched nonmergers
  - `MakeMocks_v2`
    Takes input images from SKIRT radiative transfer (run by Dr. Xuejian Shen) and creates mock HST images
  - `CutoutBackgroundsLockMethod` and `AddBackgroundtoMock`
    Adds cutouts from HST CANDELS COSMOS mosaics to mock images
  - `PlotandDivideMocks`
    Splits mock images into training, validation, and test sets
  - `PreproccesData`
    Normalizes mock images to get them ready for a CNN input

- **Convolutional Neural Network**:  
  - `BinaryMergerDataset`
    Custom dataloader and data augmentation setup
  - `ConvNano.py`  
    Train Zoobot ConvNeXT-nano CNN

- **Testing Scripts**:  
  - `TestSetAnalysis.ipynb`  
    Standard model evaluation, plus UMAP, tsne, isomap, and calibration curves.
  - `TestSetGradCAM.py`  
    Script for creating GradCAM images on the test set.

### Code Authors

- Aimee Schechter, Becky Nevin, Jacob Shen, Aaron Stemo, with help from Alex Ćiprijanović and Marina Dunn

## Citation

- [ADS](https://ui.adsabs.harvard.edu/abs/2025arXiv251115006S/abstract)
- [arXiv](https://arxiv.org/abs/2511.15006)
