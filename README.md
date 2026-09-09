# Deep Learning on EuroSAT

Land-use / land-cover classification on the **EuroSAT** satellite imagery dataset,
exploring supervised, transfer, few-shot, self-supervised and parameter-efficient
approaches in PyTorch.

Course project for Deep Learning (University of Athens, Dept. of Informatics &
Telecommunications). The original assignment brief is included as `assignment.pdf`.

## Contents (`eurosat_deep_learning.ipynb`)

1. **CNN from scratch** - a custom convolutional classifier trained on EuroSAT, with a full training loop, weight decay, and evaluation via accuracy and macro-F1.
2. **CLIP zero-shot & few-shot**
   - *Zero-shot:* classify images using CLIP text embeddings of per-class prompts.
   - *Few-shot / prototypes:* extract CLIP image embeddings, build class prototypes, and classify test images by nearest prototype (Euclidean).
3. **SimCLR self-supervised pretraining** - contrastive representation learning with paired augmentations and the NT-Xent loss, then a linear probe on the frozen encoder.
4. **LoRA fine-tuning** - parameter-efficient adaptation of a pretrained backbone with Hugging Face `peft`, compared against the baselines above.

Metrics (accuracy, precision, recall, F1) are logged with `torchmetrics` and TensorBoard; confusion matrices are plotted with seaborn.

## Running

Written for Google Colab (GPU). Locally:

```bash
pip install torch torchvision torchmetrics transformers peft scikit-learn seaborn matplotlib numpy tensorboard
```

`torchvision.datasets.EuroSAT` downloads the dataset on first use.

## Tech

Python, PyTorch, torchvision, Hugging Face (transformers, PEFT), CLIP, torchmetrics, TensorBoard, scikit-learn.
