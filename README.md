# MRNet_for_vehicle_reID

#Training
python train_test.py -s veri -t veri --optim amsgrad --lr 0.0001 --max-epoch 60 --stepsize 20 40 --train-batch-size 60 --test-batch-size 100 -a resnet50 --save-dir log/train --gpu-devices 0

#Test
python train_test.py -s veri -t veri --test-batch-size 100 --evaluate -a resnet50 --load-weights model_dir --gpu-devices 0
