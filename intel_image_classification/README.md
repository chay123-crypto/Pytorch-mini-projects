# Intel Image Classification (PyTorch)

Classifying natural scene images into 6 classes: **buildings, forest, glacier, mountain, sea, street**.
I compared a custom CNN trained from scratch against fine-tuned pretrained models (EfficientNet-B0, ConvNeXt-Tiny) and an ensemble of the two.

**Notebook:** [`intel_img.ipynb`](intel_img.ipynb) (built and run on Kaggle, Tesla T4 GPU)

## Dataset

[Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) on Kaggle.

- 14,034 training images and 3,000 test images
- 150×150 RGB, 6 classes (about 2,200 to 2,500 training images per class)
- Not included in this repo; add it to a Kaggle notebook to run the code

## Approach

| Step | Details |
|---|---|
| Preprocessing | Resize to 224×224, ImageNet mean/std normalization |
| Augmentation (train only) | Color jitter, horizontal flip, ±10° rotation |
| Custom CNN | 4 conv blocks (Conv → BatchNorm → ReLU → MaxPool), channels 32→256, global average pooling, dropout 0.3. Adam, cosine LR schedule |
| EfficientNet-B0 | ImageNet weights, new 6-class head, all layers fine-tuned (AdamW) |
| ConvNeXt-Tiny | ImageNet weights, new 6-class head, AdamW (lr 5e-5, weight decay 0.05), cosine schedule, 4 epochs, mixed precision (fp16) |
| Ensemble | Weighted average of softmax probabilities of EfficientNet and ConvNeXt |
| TTA | 8-view test-time augmentation (flips and crops) |

## Results

Evaluated on the 3,000-image test set.

| Model | Test accuracy |
|---|---|
| Custom CNN (from scratch) | ~89% |
| EfficientNet-B0 (fine-tuned) | ~94.2% |
| ConvNeXt-Tiny (fine-tuned) | **95.23%** |
| Ensemble (EfficientNet 0.3 + ConvNeXt 0.7) | [XX.XX]% |

**Takeaways**

- Transfer learning added about 5 points over the scratch CNN.
- ConvNeXt-Tiny was the best single model.
- The ensemble gained only a few tenths of a point, which is within the noise of a 3,000-image test set (roughly ±0.4%).
- Crop-based TTA did **not** help; it was slightly worse in every setting I tried.

### Error analysis (ConvNeXt-Tiny)

| Class | Precision | Recall |
|---|---|---|
| buildings | 0.97 | 0.95 |
| forest | 1.00 | 0.99 |
| glacier | 0.93 | 0.90 |
| mountain | 0.92 | 0.92 |
| sea | 0.96 | 1.00 |
| street | 0.95 | 0.97 |

Most mistakes are **glacier ↔ mountain** (42 glaciers predicted as mountain, 36 mountains predicted as glacier), followed by **buildings ↔ street**. Some of these images are ambiguous or arguably mislabeled, which limits the achievable accuracy. Published results on this dataset are roughly in the 95-96% range.

## Limitations

- The ensemble weight was chosen using the test set, so the ensemble figure is slightly optimistic. The single-model ConvNeXt result is the cleaner headline number.
- No separate validation split was used. A better setup would tune on a validation set and evaluate on the test set once.

## Possible next steps

- Hold out a validation set for tuning
- Try larger backbones (ConvNeXt-Small, EfficientNet-B2)
- Serve the model as an API (FastAPI) with weights stored in S3

## Tech stack

PyTorch, torchvision, scikit-learn, Matplotlib, Kaggle (Tesla T4)
