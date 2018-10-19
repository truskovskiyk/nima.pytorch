#!/usr/bin/env bash


# resnet18
python nima/cli.py train-model --path_to_save_csv ~/data/AVA/DATA/ava/ --path_to_images ~/data/AVA/DATA/images/ --experiment_dir resnet18-run1 --base_model resnet18 --batch_size 128 --num_epoch 15
python nima/cli.py train-model --path_to_save_csv ~/data/AVA/DATA/ava/ --path_to_images ~/data/AVA/DATA/images/ --experiment_dir resnet18-run2 --base_model resnet18 --batch_size 128 --num_epoch 15
python nima/cli.py train-model --path_to_save_csv ~/data/AVA/DATA/ava/ --path_to_images ~/data/AVA/DATA/images/ --experiment_dir resnet18-run3 --base_model resnet18 --batch_size 128 --num_epoch 15

# resnet34
python nima/cli.py train-model --path_to_save_csv ~/data/AVA/DATA/ava/ --path_to_images ~/data/AVA/DATA/images/ --experiment_dir resnet34-run1 --base_model resnet34 --batch_size 128 --num_epoch 15
python nima/cli.py train-model --path_to_save_csv ~/data/AVA/DATA/ava/ --path_to_images ~/data/AVA/DATA/images/ --experiment_dir resnet34-run2 --base_model resnet34 --batch_size 128 --num_epoch 15
python nima/cli.py train-model --path_to_save_csv ~/data/AVA/DATA/ava/ --path_to_images ~/data/AVA/DATA/images/ --experiment_dir resnet34-run3 --base_model resnet34 --batch_size 128 --num_epoch 15

# resnet50
python nima/cli.py train-model --path_to_save_csv ~/data/AVA/DATA/ava/ --path_to_images ~/data/AVA/DATA/images/ --experiment_dir resnet50-run1 --base_model resnet50 --batch_size 64 --num_epoch 10
python nima/cli.py train-model --path_to_save_csv ~/data/AVA/DATA/ava/ --path_to_images ~/data/AVA/DATA/images/ --experiment_dir resnet50-run2 --base_model resnet50 --batch_size 64 --num_epoch 10
python nima/cli.py train-model --path_to_save_csv ~/data/AVA/DATA/ava/ --path_to_images ~/data/AVA/DATA/images/ --experiment_dir resnet50-run3 --base_model resnet50 --batch_size 64 --num_epoch 10

# resnet101
python nima/cli.py train-model --path_to_save_csv ~/data/AVA/DATA/ava/ --path_to_images ~/data/AVA/DATA/images/ --experiment_dir resnet101-run1 --base_model resnet101 --batch_size 32 --num_epoch 7
python nima/cli.py train-model --path_to_save_csv ~/data/AVA/DATA/ava/ --path_to_images ~/data/AVA/DATA/images/ --experiment_dir resnet101-run2 --base_model resnet101 --batch_size 32 --num_epoch 7
python nima/cli.py train-model --path_to_save_csv ~/data/AVA/DATA/ava/ --path_to_images ~/data/AVA/DATA/images/ --experiment_dir resnet101-run3 --base_model resnet101 --batch_size 32 --num_epoch 7

# resnet152
python nima/cli.py train-model --path_to_save_csv ~/data/AVA/DATA/ava/ --path_to_images ~/data/AVA/DATA/images/ --experiment_dir resnet152-run1 --base_model resnet152 --batch_size 32 --num_epoch 7
python nima/cli.py train-model --path_to_save_csv ~/data/AVA/DATA/ava/ --path_to_images ~/data/AVA/DATA/images/ --experiment_dir resnet152-run2 --base_model resnet152 --batch_size 32 --num_epoch 7
python nima/cli.py train-model --path_to_save_csv ~/data/AVA/DATA/ava/ --path_to_images ~/data/AVA/DATA/images/ --experiment_dir resnet152-run3 --base_model resnet152 --batch_size 32 --num_epoch 7