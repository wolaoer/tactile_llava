# tactile_llava


```markdown
# Tactile-Language LLaVA

This repository explores tactile-language understanding by adapting the LLaVA
multimodal framework to tactile inputs. The original visual encoder is replaced
with a tactile encoder, enabling large language models to reason about tactile
properties and generate natural language descriptions from touch data.

## Overview

- **Modality**: Tactile + Language (optional Visual reference)
- **Backbone**: LLaVA-v1.5-7B
- **Tactile Encoder**: CLIP ViT-L/14
- **Dataset**: Touch and Go (AnyTouch subset)
- **Training Strategy**: Two-stage training
  - Stage I: Tactile–Language alignment via a projection module
  - Stage II: Instruction tuning with LoRA

## Key Features

- Reuses the LLaVA architecture for tactile-language modeling
- Parameter-efficient fine-tuning with LoRA
- Supports open-ended tactile description generation
- Qualitative and quantitative evaluation on real tactile data

## Repository Structure

```

.
├── data/               # Tactile datasets and preprocessing scripts
├── model/              # Tactile encoder and projection modules
├── train/              # Stage I and Stage II training scripts
├── eval/               # Evaluation and case study examples
├── figures/            # Model architecture and result visualizations
└── README.md

```

## Results

The model demonstrates the ability to capture key tactile attributes such as
roughness, hardness, and surface texture. Qualitative case studies show coherent
tactile descriptions, while quantitative evaluation indicates effective
tactile-language alignment after limited training.

## Notes

This project is intended as a technical exploration rather than a
state-of-the-art benchmark. The focus is on validating the feasibility of
tactile-language modeling using existing multimodal LLM architectures.

## Acknowledgements

This work builds upon LLaVA, CLIP, and the Touch and Go / AnyTouch datasets.
```




