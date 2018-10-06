# LightNER

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/LightNER.svg)](https://badge.fury.io/py/LightNER)
<!-- [![Documentation Status](https://readthedocs.org/projects/tensorboard-wrapper/badge/?version=latest)](http://tensorboard-wrapper.readthedocs.io/en/latest/?badge=latest) -->
<!-- [![Downloads](https://pepy.tech/badge/torch-scope)](https://pepy.tech/project/LightNER) -->

**Check Our New NER Toolkit🚀🚀🚀**
- **Inference**:
  - **[LightNER](https://github.com/LiyuanLucasLiu/LightNER)**: inference w. models pre-trained / trained w. *any* following tools, *efficiently*. 
- **Training**:
  - **[LD-Net](https://github.com/LiyuanLucasLiu/LD-Net)**: train NER models w. efficient contextualized representations.
  - **[VanillaNER](https://github.com/LiyuanLucasLiu/Vanilla_NER)**: train vanilla NER models w. pre-trained embedding.
- **Distant Training**:
  - **[AutoNER](https://shangjingbo1226.github.io/AutoNER/)**: train NER models w.o. line-by-line annotations and get competitive performance.

--------------------------------

This package supports to conduct inference with models pre-trained by:
- [Vanilla_NER](https://github.com/LiyuanLucasLiu/Vanilla_NER): vanilla sequence labeling models.
- [LD-Net](https://github.com/LiyuanLucasLiu/LD-Net): sequence labeling models w. efficient contextualized representation.
- [AutoNER](https://github.com/shangjingbo1226/AutoNER): distant supervised named entity recognition models (*no line-by-line annotations for training*).

We are in an early-release beta. Expect some adventures and rough edges.

## Quick Links

- [Installation](#installation)
- [Usage](#usage)

## Installation

To install via pypi:
```
pip install lightner
```

To build from source:
```
pip install git+https://github.com/LiyuanLucasLiu/LightNER
```
or
```
git clone https://github.com/LiyuanLucasLiu/LightNER.git
cd LightNER
python setup.py install
```

## Usage

### Pre-trained Models

|               | Model             | Task            | Performance            |
| ------------- |-------------      | -------------   | -------------          |
| LD-Net        | [pner0.th](http://dmserv4.cs.illinois.edu/pner1.th) | NER for (PER, LOC, ORG & MISC) | F1 92.21 |
| LD-Net        | [pnp0.th](http://dmserv4.cs.illinois.edu/pnp0.th)   | Chunking                       | F1 95.79 |  
| Vanilla_NER   |                                                               | NER for (PER, LOC, ORG & MISC) | |
| Vanilla_NER   |                                                               | Chunking                       | |
| AutoNER       |                                                               | Distant NER trained w.o. line-by-line annotations | |


### Decode API

The decode api can be called in the following way:
```
from lightner import decoder_wrapper
model = decoder_wrapper()
model.decode(["Ronaldo", "won", "'t", "score", "more", "than", "30", "goals", "for", "Juve", "."])
```

The ```decode()``` method also can conduct decoding at document level (takes list of list of ```str``` as input) or corpus level (takes list of list of list of ```str``` as input).

The ```decoder_wrapper``` method can be customized by choosing a different pre-trained model or passing an additional ```configs``` file as:
```
model = decoder_wrapper(URL_OR_PATH_TO_CHECKPOINT, configs)
```
And you can access the config options by:
```
lightner decode -h
```

### Console

After installing and downloading the pre-trained mdoels, conduct the inference by 
```
lightner decode -m MODEL_FILE -i INPUT_FILE -o OUTPUT_FILE
```

You can find more options by:
```
lightner decode -h
```

The current accepted paper format is as below (tokenized by line break and ```-DOCSTART-``` is optional):
```
-DOCSTART-

Ronaldo
won
't
score
more
30
goals
for
Juve
.
```

The output would be:
```
<PER> Ronaldo </PER> won 't score more than 30 goals for <ORG> Juve </ORG> . 
```
