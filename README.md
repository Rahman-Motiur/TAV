# TAV PyTorch

PyTorch implementation of **Text-Assisted Vision Model for Medical Image Segmentation**.

TAV improves medical image segmentation by using both image features and corresponding medical text annotations. The core components are:

- **Tri-Guided Attention Module (TGAM)**: computes visual-visual, language-language, and language-visual attention.
- **Attention Gate (AG)**: regulates the influence of TGAM so attention maps enrich visual features without overwhelming them.
- **Joint image-text fusion**: projects text features into each visual encoder stage for early cross-modal fusion.
- **Contrastive alignment loss**: aligns visual and text embeddings in a shared latent space.
- **Decoder segmentation head**: reconstructs dense segmentation masks from fused multi-scale features.

This repository includes a compact runnable implementation. It uses a lightweight text encoder for demos and can be extended with BERT, ClinicalBERT, BioBERT, or other medical language encoders.

## Repository Layout

```text
tav-pytorch/
  tav/
    data/              CSV image-mask-text dataset loader
    models/            TAV, TGAM, attention gate, encoders, decoder
    training/          Segmentation and contrastive losses, metrics
    utils/             Reproducibility helpers
  configs/             Example experiment configs
  scripts/             Demo and training commands
  tests/               Forward and loss tests
```

## Installation

```bash
git clone https://github.com/<your-user>/TAV.git
cd TAV
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Demo Forward Pass

```bash
python scripts/demo_tav_forward.py
```

## Training

Prepare a CSV file:

```csv
image,mask,text
data/images/case_001.npy,data/masks/case_001.npy,"bilateral lung infection in upper left and upper right lung"
```

Run:

```bash
python scripts/train_tav.py --config configs/tav_mosmeddata.yaml
```

## Paper-to-Code Mapping

| Paper component | Code |
| --- | --- |
| Visual encoder stages | `tav/models/vision_encoder.py` |
| Text encoder and projections | `tav/models/text_encoder.py` |
| TGAM | `tav/models/tgam.py` |
| Attention Gate | `tav/models/attention_gate.py` |
| TAV forward workflow | `tav/models/tav.py` |
| Contrastive + Dice + CE loss | `tav/training/losses.py` |

## Notes

The paper evaluates TAV with pretrained text encoders such as BERT variants. This implementation keeps the interface simple and runnable by default. To reproduce paper-scale results, replace `LightweightTextEncoder` with a pretrained medical text encoder and keep the same TGAM interface.
