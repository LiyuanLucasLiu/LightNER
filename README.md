# LightNER

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/torch-scope.svg)](https://badge.fury.io/py/LightNER)
<!-- [![Documentation Status](https://readthedocs.org/projects/tensorboard-wrapper/badge/?version=latest)](http://tensorboard-wrapper.readthedocs.io/en/latest/?badge=latest) -->
<!-- [![Downloads](https://pepy.tech/badge/torch-scope)](https://pepy.tech/project/LightNER) -->

A Toolkit to conduct inference with models pre-trained by LD-Net / AutoNER / VanillaNER / ...

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

### Decode API

The decode api can be called in the following way:
```
from lightner import decoder_wrapper
model = decoder_wrapper(PATH_TO_CHECKPOINT)
model.decode(["Ronaldo", "won", "'t", "score", "more", "than", "30", "goals", "for", "Juve", "."])
```

The ```decode()``` method also can conduct decoding at document level (takes list of list of ```str``` as input) or corpus level (takes list of list of list of ```str``` as input).

The ```decoder_wrapper``` method can be customized by passing an additional ```configs``` file as:
```
model = decoder_wrapper(PATH_TO_CHECKPOINT, configs)
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
