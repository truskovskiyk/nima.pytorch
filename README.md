# PyTorch NIMA: Neural IMage Assessment

PyTorch implementation of [Neural IMage Assessment](https://arxiv.org/abs/1709.05424) by Hossein Talebi and Peyman Milanfar. You can learn more from [this post at Google Research Blog](https://research.googleblog.com/2017/12/introducing-nima-neural-image-assessment.html). 


## Installing

### Docker
```
docker run -it truskovskiyk/nima:latest /bin/bash
```

### PYPI package (In Progress)
```
pip install nima
```

### VirtualEnv
```bash
git clone https://github.com/truskovskiyk/nima.pytorch.git
cd nima.pytorch
virtualenv -p python3.7 env
source ./env/bin/activate
```


## Dataset

The model was trained on the [AVA (Aesthetic Visual Analysis) dataset](http://refbase.cvc.uab.es/files/MMP2012a.pdf)
You can get it from [here](https://github.com/mtobeiyf/ava_downloader)
Here are some examples of images with theire scores 
![result1](https://3.bp.blogspot.com/-_BuiLfAsHGE/WjgoftooRiI/AAAAAAAACR0/mB3tOfinfgA5Z7moldaLIGn92ounSOb8ACLcBGAs/s1600/image2.png)

## Pre-train model (In Progress)

```bash

```


## Deployment (In progress)

```bash

```

## Usage
```bash
nima-cli

Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  get_image_score  Get image scores
  prepare_dataset  Parse, clean and split dataset
  run_web_api      Start server for model serving
  train_model      Train model
  validate_model   Validate model
```


## Previous version of this project is still valid and works
[you can find here](https://github.com/truskovskiyk/nima.pytorch/tree/v1)

## Contributing

Contributing are welcome


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* [neural-image-assessment in keras](https://github.com/titu1994/neural-image-assessment)
* [Neural-IMage-Assessment in pytorch](https://github.com/kentsyx/Neural-IMage-Assessment)
* [pytorch-mobilenet-v2](https://github.com/tonylins/pytorch-mobilenet-v2)
* [origin NIMA article](https://arxiv.org/abs/1709.05424)
* [origin MobileNetV2 article](https://arxiv.org/pdf/1801.04381)
* [Post at Google Research Blog](https://research.googleblog.com/2017/12/introducing-nima-neural-image-assessment.html)
* [Heroku: Cloud Application Platform](https://www.heroku.com/)
