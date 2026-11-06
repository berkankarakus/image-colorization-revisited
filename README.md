# Image Colorization GAN Revisited

This repository contains a revisited and simplified version of an older image-colorization project.

The original project had several issues that could prevent training or make the results unstable:

- the CIFAR-10 data was 32×32 while the model expected 128×128 inputs;
- the final preview referenced an undefined variable (`plt_a`);
- the optimizer used the deprecated `lr` argument;
- cosine similarity was trained with an unsuitable binary-cross-entropy loss;
- previews were not shown or saved at epoch boundaries; and
- training always attempted to authenticate with Google Drive.

The repaired version keeps the original conditional GAN idea and CIFAR-10 dataset, but trains at the native 32×32 resolution. Upscaling CIFAR-10 would not add image information. It saves and displays a small preview grid after every epoch and stores local checkpoints without requiring Google Drive authentication.

## Run in Google Colab

Upload `image_colorization.py` to Colab and run it from a notebook cell:

```python
%run image_colorization.py
```

The script downloads CIFAR-10 through Keras on the first run. Epoch previews are written to `colorization_outputs/` and checkpoints to `colorization_outputs/weights/`.

The script is intended for an interactive Colab notebook. If it is launched with `!python`, previews are still saved but matplotlib figures are not rendered inline.

This minimal version uses the ordinary Keras training loop. Selecting a TPU in Colab does not by itself distribute this loop across TPU replicas; use a GPU or CPU for this version unless you add an explicit `tf.distribute.TPUStrategy` implementation.

## Model outputs

Generated images are intentionally not included in this minimal repository. They are experiment artifacts, change between runs, and can make the repository unnecessarily large. The script generates them locally after each epoch. A selected preview can be added later if the repository needs a visual example.

## Scope

This is an educational repair of the original experiment, not a high-resolution colorization system. CIFAR-10 is a small 32×32 benchmark dataset, so the results should not be interpreted as photo-realistic colorization.
