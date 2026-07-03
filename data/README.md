# Data Preparation

This folder contains the datasets used in our experiments.

We use four datasets:

- WaterBirds
- CelebA
- Texture
- PACS

## WaterBirds

The generated WaterBirds dataset is publicly available from Dunlap et al. (2023), and we directly use their released version.

Please refer to:

- GitHub: [ALIA: Augmenting with Language-guided Image Augmentation](https://github.com/lisadunlap/ALIA)
- Weights & Biases: [https://wandb.ai/clipinvariance/ALIA](https://wandb.ai/clipinvariance/ALIA)

## CelebA, Texture, and PACS

The remaining three datasets, CelebA, Texture, and PACS, are not publicly available. We therefore generate them ourselves using the code provided by Yuan et al. (2024):

[https://github.com/YuanJianhao508/NotJustPrettyPictures](https://github.com/YuanJianhao508/NotJustPrettyPictures)

Following Yuan et al. (2024), we use dataset-specific textual prompts for image-conditioned editing with Stable Diffusion v1.5.

The generated training data should be placed in `train_intervene/` for CelebA and Texture.  
For PACS, the generated training data should be placed in the corresponding `*_to_*` folders.

The original training data should be placed in `train/`.

## Generation Prompts

### PACS

For PACS, we use the prompt:

```text
A [DOMAIN] of [CLASS LABEL]
```

where `DOMAIN` denotes the target PACS domain:

```text
Art Painting, Cartoon, Photo, Sketch
```

### CelebA

For CelebA, prompts are designed to preserve the target label, hair color, while flipping the spurious attribute, gender.

Specifically, images labeled as blonde are edited using the prompt:

```text
blonde male
```

Images labeled as non-blonde are edited using the prompt:

```text
non-blonde female
```

### Texture

For Texture, we use the prompt:

```text
[STYLE] [CLASS LABEL]
```

where `STYLE` is sampled from the following set:

```text
pointillism, rubin statue, rusty statue, ceramic, vaporwave, stained glass,
wood statue, metal statue, bronze statue, iron statue, marble statue,
stone statue, mosaic, furry, corel draw, simple sketch, stroke drawing,
black ink painting, silhouette painting, black pen sketch, quickdraw sketch,
grainy, surreal art, oil painting, fresco, naturalistic painting, stylised
```

## Folder Structure

The `data/` folder should be organized as follows:

```text
data/
├── waterbirds/
├── celeba/
├── texture/
└── pacs/
```

For `celeba/` and `texture/`, the folder structure is:

```text
celeba/
├── test/
├── train/
├── train_intervene/
└── val/

texture/
├── test/
├── train/
├── train_intervene/
└── val/
```

Here, `train/` contains the original training images, and `train_intervene/` contains the generated training images.

For PACS, the folder structure is:

```text
pacs/
├── art_painting/
├── cartoon/
├── photo/
└── sketch/
```

Each PACS domain folder contains the original train/validation/test splits and the generated training data folders:

```text
pacs/
├── art_painting/
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── cartoon_to_art_painting/
│   ├── cartoon_to_photo/
│   └── cartoon_to_sketch/
├── cartoon/
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── cartoon_to_art_painting/
│   ├── cartoon_to_photo/
│   └── cartoon_to_sketch/
├── photo/
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── cartoon_to_art_painting/
│   ├── cartoon_to_photo/
│   └── cartoon_to_sketch/
└── sketch/
    ├── train/
    ├── val/
    ├── test/
    ├── cartoon_to_art_painting/
    ├── cartoon_to_photo/
    └── cartoon_to_sketch/
```

The images in each split are further organized by class names. For example:

```text
pacs/
└── art_painting/
    └── train/
        ├── dog/
        ├── elephant/
        ├── giraffe/
        ├── guitar/
        ├── horse/
        ├── house/
        └── person/
```

Similarly, generated PACS images are also organized by class names:

```text
pacs/
└── art_painting/
    └── cartoon_to_art_painting/
        ├── dog/
        ├── elephant/
        ├── giraffe/
        ├── guitar/
        ├── horse/
        ├── house/
        └── person/
```
