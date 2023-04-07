# Wealth Alpaca-LoRa

This is an implementation of Alpaca-LoRa that uses a custom variant of the Alpaca dataset focused on Wealth/Finance. The training is done using [Kaggle](https://www.kaggle.com)'s free GPUs (2x Tesla T4s).

## Data Generation

### Installation

To install the required packages, run:

`pip install -r requirements.txt`


### Usage

To generate data, run:

`python -m generate_data generate_instructions_from_websites`


This will generate data from the sources listed in `sources.txt`.

### Custom Sources

You can add your own sources to `sources.txt`.


Make sure to run `python -m generate_data` again after adding new sources to `sources.txt`.

---
## Dataset
More details on the dataset can be found [here](https://huggingface.co/datasets/gbharti/wealth-alpaca_lora)

---

## Training

The training script `wealth-alpaca_lora.py` is optimized for training on Kaggle using 2x Tesla T4s and can be run through:

```!torchrun --nproc_per_node=2 wealth-alpaca_lora.py``` 

Training notebook on Kaggle can be found here: https://www.kaggle.com/code/gbhacker23/wealth-alpaca-lora

---
## Inference
LoRa Weights: https://huggingface.co/gbharti/finance-alpaca-lora

Inference: https://colab.research.google.com/drive/1lIlJpsMq4JP1GszUw_oZhyKr2irsMcq_?usp=sharing


Interface for inference: https://colab.research.google.com/drive/11V_w7y5hEbVgsx9we8RMyapY9bEeuY5X?usp=sharing

---

## Performance analysis (Work in Progress)
Comparison between Alpaca-LoRa and Wealth Alpaca-LoRa: https://docs.google.com/document/d/1ldso1tmFLkg0flePdDKj_iTc8gcmdRtoe9n1dYN1W10/edit?usp=sharing

ChatGPT responses: https://docs.google.com/document/d/1PevbI89jgiPYaSSCnVcLIF9ObiDbAf_LsmsTx-SIvBg/edit?usp=sharing