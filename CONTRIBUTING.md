# Contributing

Useful contribution areas:

- BERT, BioBERT, ClinicalBERT, and MedGPT text encoder adapters
- MosMedData+ and QaTa-COV19 preprocessing scripts
- text granularity experiments
- attention heatmap visualization
- mixed-precision and distributed training

Before opening a pull request, run:

```bash
python scripts/demo_tav_forward.py
pytest tests
```
