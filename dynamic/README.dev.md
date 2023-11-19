# dev

```sh
HSA_OVERRIDE_GFX_VERSION=10.3.0 python run.py segmentation --model_name deeplabv3_resnet101

HSA_OVERRIDE_GFX_VERSION=10.3.0 python run.py segmentation --model_name deeplabv3_resnet50 --seed 10 --num_epochs 1 --run_distill --run_test --pretrained
```
