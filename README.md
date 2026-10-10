# GAHR

Official implementation of **Rethinking Multimodal Fake News Detection in the Generative AI Era**.


## Installation

```bash
pip install -r requirements.txt
```

The pretrained backbones are downloaded from Hugging Face on their first use.


## Data format

Organize the data as follows:

```text
data/
├── train.csv
├── val.csv
├── test.csv
└── images/
```

Each available split CSV must contain `text` and `label` columns.

The optional `category` column enables GAHR's generative-modality auxiliary loss. Supported values are `NRN`, `NFN`, `ICRN`, `ICFN`, `TCRN`, and `TCFN` (or their descriptive names).


## Run

First extract frozen backbone features:

```bash
python data_processing.py --data_dir data --image_dir images --feature_dir features --language zh
```

Then train and evaluate GAHR:

```bash
python main.py --feature_dir features --output_dir outputs --mode train_test
```

For English backbones, pass `--language en` to both commands. The feature dimension settings in `config.py` correspond to the supplied backbone choices; update them together if you substitute a backbone.


## Outputs

Each run creates a timestamped directory under `outputs/logs/` containing epoch logs, metrics, and plain prediction files. The best checkpoint is stored at `outputs/model/best_model.pth`. 


## Repository layout

```text
config.py           Default backbones and optimization settings
data_processing.py  Raw CSV/image input to frozen features
dataset.py          Feature dataset and auxiliary-target handling
model.py            GAHR model and loss
main.py             Training and evaluation entry point
utils.py            Metrics, checkpoints, and logging
```


## Citation

If you use this code, please cite:

```bibtex
@article{shen2026rethinking,
  title={Rethinking Multimodal Fake News Detection in the Generative AI Era},
  author={Shen, Wenbin and Qin, Guoxuan and Yao, Guangxu and Wang, Baodong and Rui, Yuanbo and Lian, Zhichao},
  journal={arXiv preprint arXiv:2609.36850},
  year={2026}
}
```
