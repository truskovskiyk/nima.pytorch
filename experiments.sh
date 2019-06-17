#!/usr/bin/env bash


# resnet18
neuro submit --no-wait-start -n nima-train1 -g 1 -c 5 -m 12G --gpu-model nvidia-tesla-k80 --http 8080 --non-preemptible -v storage://truskovskiyk/common/nima-datasets/DATA/:/data:rw truskovskyi/nima:latest "python nima/cli.py train-model --batch_size 128 --model_type resnet18 --path_to_save_csv /data/ --path_to_images /data/images/ --experiment_dir /data/exp/t1-resnet18 --num_epoch 100"
neuro submit --no-wait-start -n nima-train2 -g 1 -c 5 -m 12G --gpu-model nvidia-tesla-k80 --http 8080 --non-preemptible -v storage://truskovskiyk/common/nima-datasets/DATA/:/data:rw truskovskyi/nima:latest "python nima/cli.py train-model --batch_size 128 --model_type resnet18 --path_to_save_csv /data/ --path_to_images /data/images/ --experiment_dir /data/exp/t2-resnet18 --num_epoch 100"
neuro submit --no-wait-start -n nima-train3 -g 1 -c 5 -m 12G --gpu-model nvidia-tesla-k80 --http 8080 --non-preemptible -v storage://truskovskiyk/common/nima-datasets/DATA/:/data:rw truskovskyi/nima:latest "python nima/cli.py train-model --batch_size 128 --model_type resnet18 --path_to_save_csv /data/ --path_to_images /data/images/ --experiment_dir /data/exp/t3-resnet18 --num_epoch 100"

# resnet34
neuro submit --no-wait-start -n nima-train4 -g 1 -c 5 -m 12G --gpu-model nvidia-tesla-k80 --http 8080 --non-preemptible -v storage://truskovskiyk/common/nima-datasets/DATA/:/data:rw truskovskyi/nima:latest "python nima/cli.py train-model --batch_size 128 --model_type resnet34 --path_to_save_csv /data/ --path_to_images /data/images/ --experiment_dir /data/exp/t1-resnet34 --num_epoch 100"
neuro submit --no-wait-start -n nima-train5 -g 1 -c 5 -m 12G --gpu-model nvidia-tesla-k80 --http 8080 --non-preemptible -v storage://truskovskiyk/common/nima-datasets/DATA/:/data:rw truskovskyi/nima:latest "python nima/cli.py train-model --batch_size 128 --model_type resnet34 --path_to_save_csv /data/ --path_to_images /data/images/ --experiment_dir /data/exp/t2-resnet34 --num_epoch 100"
neuro submit --no-wait-start -n nima-train6 -g 1 -c 5 -m 12G --gpu-model nvidia-tesla-k80 --http 8080 --non-preemptible -v storage://truskovskiyk/common/nima-datasets/DATA/:/data:rw truskovskyi/nima:latest "python nima/cli.py train-model --batch_size 128 --model_type resnet34 --path_to_save_csv /data/ --path_to_images /data/images/ --experiment_dir /data/exp/t3-resnet34 --num_epoch 100"

# resnet50
neuro submit --no-wait-start -n nima-train7 -g 1 -c 5 -m 12G --gpu-model nvidia-tesla-k80 --http 8080 --non-preemptible -v storage://truskovskiyk/common/nima-datasets/DATA/:/data:rw truskovskyi/nima:latest "python nima/cli.py train-model --batch_size 64 --model_type resnet50 --path_to_save_csv /data/ --path_to_images /data/images/ --experiment_dir /data/exp/t1-resnet50 --num_epoch 100"
neuro submit --no-wait-start -n nima-train8 -g 1 -c 5 -m 12G --gpu-model nvidia-tesla-k80 --http 8080 --non-preemptible -v storage://truskovskiyk/common/nima-datasets/DATA/:/data:rw truskovskyi/nima:latest "python nima/cli.py train-model --batch_size 64 --model_type resnet50 --path_to_save_csv /data/ --path_to_images /data/images/ --experiment_dir /data/exp/t2-resnet50 --num_epoch 100"
neuro submit --no-wait-start -n nima-train9 -g 1 -c 5 -m 12G --gpu-model nvidia-tesla-k80 --http 8080 --non-preemptible -v storage://truskovskiyk/common/nima-datasets/DATA/:/data:rw truskovskyi/nima:latest "python nima/cli.py train-model --batch_size 64 --model_type resnet50 --path_to_save_csv /data/ --path_to_images /data/images/ --experiment_dir /data/exp/t3-resnet50 --num_epoch 100"

# resnet101
neuro submit --no-wait-start -n nima-train10 -g 1 -c 5 -m 12G --gpu-model nvidia-tesla-k80 --http 8080 --non-preemptible -v storage://truskovskiyk/common/nima-datasets/DATA/:/data:rw truskovskyi/nima:latest "python nima/cli.py train-model --batch_size 32 --model_type resnet101 --path_to_save_csv /data/ --path_to_images /data/images/ --experiment_dir /data/exp/t1-resnet101 --num_epoch 100"
neuro submit --no-wait-start -n nima-train11 -g 1 -c 5 -m 12G --gpu-model nvidia-tesla-k80 --http 8080 --non-preemptible -v storage://truskovskiyk/common/nima-datasets/DATA/:/data:rw truskovskyi/nima:latest "python nima/cli.py train-model --batch_size 32 --model_type resnet101 --path_to_save_csv /data/ --path_to_images /data/images/ --experiment_dir /data/exp/t2-resnet101 --num_epoch 100"
neuro submit --no-wait-start -n nima-train12 -g 1 -c 5 -m 12G --gpu-model nvidia-tesla-k80 --http 8080 --non-preemptible -v storage://truskovskiyk/common/nima-datasets/DATA/:/data:rw truskovskyi/nima:latest "python nima/cli.py train-model --batch_size 32 --model_type resnet101 --path_to_save_csv /data/ --path_to_images /data/images/ --experiment_dir /data/exp/t3-resnet101 --num_epoch 100"


neuro submit -n nima-tf -g 0 -c 2 -m 4G --http 8080 --non-preemptible -v storage://truskovskiyk/common/nima-datasets/DATA/:/data:rw tensorflow/tensorflow:latest "tensorboard --logdir /data/exp/ --port 8080"
