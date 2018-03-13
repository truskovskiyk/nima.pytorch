# PyTorch NIMA: Neural IMage Assessment

PyTorch implementation of [Neural IMage Assessment](https://arxiv.org/abs/1709.05424) by Hossein Talebi and Peyman Milanfar. You can learn more from [this post at Google Research Blog](https://research.googleblog.com/2017/12/introducing-nima-neural-image-assessment.html). 


## Installing

```bash
git clone https://github.com/truskovskiyk/nima.pytorch.git 
cd nima.pytorch
virtualenv -p python3.6 env
source ./env/bin/activate
pip install -r requirements/linux_gpu.txt
```

or You can just use ready [Dockerfile](./Dockerfile)


## Dataset

The model was trained on the [AVA (Aesthetic Visual Analysis) dataset](http://refbase.cvc.uab.es/files/MMP2012a.pdf)
You can get it from [here](https://github.com/mtobeiyf/ava_downloader)
Here are some examples of images with theire scores 
![result1](https://3.bp.blogspot.com/-_BuiLfAsHGE/WjgoftooRiI/AAAAAAAACR0/mB3tOfinfgA5Z7moldaLIGn92ounSOb8ACLcBGAs/s1600/image2.png)

## Model 

Used MobileNetV2 architecture as described in the paper [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/pdf/1801.04381).

## Pre-train model  

You can use this [pretrain-model](https://s3-us-west-1.amazonaws.com/models-nima/pretrain-model.pth) with
```bash
val_emd_loss = 0.079
test_emd_loss = 0.080
```
## Deployment

Deployed model on [heroku](https://www.heroku.com/) URL is https://neural-image-assessment.herokuapp.com/ You can use it for testing in Your own images, but pay attention, that's free service, so it cannot handel too many requests. Here is simple curl command to test deployment models
```bash
curl  -X POST -F "file=@123.jpg" https://neural-image-assessment.herokuapp.com/api/get_scores
```
Please use our [swagger](https://neural-image-assessment.herokuapp.com/apidocs) for interactive testing 


## Usage

Clean and prepare dataset
```bash
export PYTHONPATH=.
python nima/cli.py prepare_dataset --path_to_ava_txt ./DATA/ava/AVA.txt \
                                    --path_to_save_csv ./DATA/ava \
                                    --path_to_images ./DATA/images/

```

Train model
```bash
export PYTHONPATH=.
python nima/cli.py train_model --path_to_save_csv ./DATA/ava/ \
                                --path_to_images ./DATA/images \
                                --batch_size 16 \
                                --num_workers 2 \
                                --num_epoch 15 \
                                --init_lr 0.009 \
                                --experiment_dir_name firts0.009


```
Use tensorboard to tracking training progress

```bash
tensorboard --logdir .
```
Validate model on val and test datasets
```bash
export PYTHONPATH=.
python nima/cli.py validate_model --path_to_model_weight ./pretrain-model.pth \
                                    --path_to_save_csv ./DATA/ava \
                                    --path_to_images ./DATA/images \
                                    --batch_size 16 \
                                    --num_workers 4
```
Get scores for one image
```bash
export PYTHONPATH=.
python nima/cli.py get_image_score --path_to_model_weight ./pretrain-model.pth \
                                       --path_to_image test_image.jpg
```
   
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
