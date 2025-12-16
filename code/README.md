# Setup

```shell
conda create -n levyjepa python=3.10 -y
conda activate levyjepa
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install scipy tensorboard timm wandb hydra-core datasets==2.14.6 huggingface-hub "pyarrow<13.0.0" "numpy<2.0"
```

# Experiment 1

Starting from the [LeJepa minimal demo](https://github.com/rbalestr-lab/lejepa/blob/main/MINIMAL.md),
I added a 2D Levy prior[^1] to the final embedding layer of the ViT whose strength was controlled by parameter `gamma`:
`gamma=0` corresponds to the LeJepa baseline.[^2]

## Baseline (without Levy 2D regularizer)

```shell
WANDB_MODE=offline python ./minimal.py +device_name=cuda +levy_alpha=1 +levy_sigma=0.125 +levy_lam=2 +gamma=0 +kappa=1 +lamb=0.02 +V=4 +proj_dim=16 +net_lr_bs256=5e-4 +bs=96 +epochs=800
wandb:
wandb: Run history:
wandb:            global_step ▁▁▁▁▂▂▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▅▅▅▅▆▆▆▆▆▆▆▆▆▇▇▇▇█
wandb:               test/acc ▁▂▃▄▄▅▆▇▇▇▇▇▇▇▇█▇▇██████████████████████
wandb:              train/inv █▇▇▇█▆▆▆▆▅▄▄▄▄▃▃▄▃▄▃▃▃▃▄▃▃▃▂▂▁▂▂▂▂▃▁▂▁▂▁
wandb:      train/lejepa_loss █▇▇▆▆▅▅▄▄▄▄▄▃▄▃▃▃▃▃▃▃▂▂▂▂▁▁▂▂▂▂▁▂▂▁▁▁▂▁▂
wandb:               train/lr ████████▇▇▆▆▆▆▆▆▅▅▅▄▄▄▃▃▃▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁
wandb: train/orig_lejepa_loss █▇▇▅▅▅▅▅▅▅▅▄▃▃▃▃▃▂▃▃▂▃▂▃▂▂▂▂▁▂▁▂▂▁▁▁▁▁▂▁
wandb:       train/probe_loss ▇█▇█▆▇▇▅▆▅▅▃▄▃▄▂▃▂▁▂▂▂▁▂▁▁▁▂▁▂▁▁▁▂▁▂▁▁▁▁
wandb:           train/sigreg █▄▄▄▃▃▃▃▃▃▂▂▃▂▃▂▃▃▃▂▂▂▂▂▂▂▁▂▂▂▂▂▁▁▁▁▂▂▁▁
wandb:           train/within ▁▂▂▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇▇▇▇▇▇▇▇█████
wandb:
wandb: Run summary:
wandb:            global_step 800
wandb:               test/acc 0.90726
wandb:              train/inv 0.03608
wandb:      train/lejepa_loss 0.05819
wandb:               train/lr 8e-05
wandb: train/orig_lejepa_loss 0.05819
wandb:       train/probe_loss 0.44093
wandb:           train/sigreg 1.14062
wandb:           train/within 1.54163
```

## With Levy 2D regularizer

```shell
WANDB_MODE=offline python ./minimal.py +device_name=cuda +levy_alpha=1 +levy_sigma=0.125 +levy_lam=2 +gamma=1 +kappa=1 +lamb=0.02 +V=4 +proj_dim=16 +net_lr_bs256=5e-4 +bs=96 +epochs=800
...
wandb:
wandb: Run history:
wandb:            global_step ▁▁▁▁▁▂▂▂▂▂▃▃▃▄▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇████
wandb:               test/acc ▁▃▅▅▄▆▇▇▇▆▇▇████████████████████████████
wandb:              train/inv ██▇▆▆▅▅▅▃▃▃▃▃▃▃▃▂▂▃▂▂▂▂▂▁▂▁▂▂▁▁▂▁▁▁▁▁▁▁▁
wandb:      train/lejepa_loss ██▇▇▅▇▆▆▅▅▄▅▃▂▂▂▂▃▃▃▃▂▁▃▂▃▂▃▂▃▂▂▂▂▂▁▂▁▃▂
wandb:               train/lr █████████▇▇▇▇▇▇▆▆▆▆▅▄▄▄▄▄▃▃▃▃▃▂▂▂▁▁▁▁▁▁▁
wandb: train/orig_lejepa_loss ▇█▆▆▄▄▃▃▃▃▃▃▃▄▃▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▁▁▁▁▁▁▁▁▁▁
wandb:       train/probe_loss █▆▆▅▅▅▃▄▅▄▃▃▄▄▃▃▃▃▂▂▂▂▂▂▃▁▂▁▂▂▂▃▁▂▁▁▂▂▂▁
wandb:           train/sigreg █▅▆▄▄▃▃▃▃▂▂▃▂▂▂▂▂▂▂▂▂▁▁▂▁▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁
wandb:           train/within █▆▃▂▂▁▁▂▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:
wandb: Run summary:
wandb:            global_step 800
wandb:               test/acc 0.91287
wandb:              train/inv 0.03697
wandb:      train/lejepa_loss 0.05773
wandb:               train/lr 8e-05
wandb: train/orig_lejepa_loss 0.05625
wandb:       train/probe_loss 0.50342
wandb:           train/sigreg 1
wandb:           train/within 0.00148
```

# Experiment 2

On the hunch that the [LeJepa minimal demo](https://github.com/rbalestr-lab/lejepa/blob/main/MINIMAL.md) is saturated, I created a parameter `kappa`
to control the strength of the invariance loss, with `kappa=0` corresponding to no invariance loss.

## Baseline (without Levy 2D regularizer, and without invariance loss)

```shell
WANDB_MODE=offline python ./minimal.py +device_name=cuda +levy_alpha=1 +levy_sigma=0.125 +levy_lam=1 +gamma=0 +kappa=0 +lamb=0.02 +V=4 +proj_dim=16 +net_lr_bs256=5e-5 +bs=96 +epochs=800
...
wandb:
wandb: Run history:
wandb:            global_step ▁▁▁▁▁▂▂▂▃▃▃▃▃▃▃▃▃▄▄▄▅▅▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇▇▇█
wandb:               test/acc ▁▃▃▄▄▄▅▅▅▅▅▆▆▆▆▆▆▆▇▆▇▇▇▇▇▇▇▇████████████
wandb:              train/inv █▇▇▇▇▅▅▅▅▃▃▃▃▃▂▃▃▃▃▂▂▂▂▂▃▂▂▂▂▂▂▂▂▁▂▂▁▁▁▂
wandb:      train/lejepa_loss ▆▅▅▇▆▄▅█▂▆▆▄▄▄▂▄▄▄▃▄▂▄▃▃▃▅▂▃▄▂▂▃▃▂▂▂▃▃▁▂
wandb:               train/lr ██████████▇▇▇▇▆▅▅▅▄▄▄▄▄▃▃▃▂▂▂▂▂▂▂▁▁▁▁▁▁▁
wandb: train/orig_lejepa_loss █▆▆▅▄▂▃▂▃▂▂▂▃▃▁▂▂▂▂▂▂▂▁▂▁▂▂▁▂▁▁▁▁▂▁▂▁▁▁▁
wandb:       train/probe_loss █▇▆▇▆▆▆▅▅▄▅▆▅▅▄▄▃▃▂▅▃▄▃▃▂▃▂▃▄▂▃▃▃▁▃▁▂▄▃▃
wandb:           train/sigreg ▆█▆▅▄▅▂▅▄▃▂▃▃▄▄▃▃▅▅▂▃▁▃▄▁▃▂▁▃▄▁▂▁▂▃▁▂▁▁▁
wandb:           train/within ██▇▄▅▄▃▁▁▂▁▃▂▁▁▂▁▂▁▂▂▂▁▂▁▂▂▁▂▂▁▂▃▁▁▂▂▁▃▂
wandb:
wandb: Run summary:
wandb:            global_step 800
wandb:               test/acc 0.56943
wandb:              train/inv 0.32771
wandb:      train/lejepa_loss 0.01282
wandb:               train/lr 1e-05
wandb: train/orig_lejepa_loss 0.33397
wandb:       train/probe_loss 1.46786
wandb:           train/sigreg 0.64062
wandb:           train/within 0.50736
wandb:
```

## with levy regularizer (but without invariance loss)
```shell
WANDB_MODE=offline python ./minimal.py +device_name=cuda +levy_alpha=1 +levy_sigma=0.125 +levy_lam=1 +gamma=1 +kappa=0 +lamb=0.02 +V=4 +proj_dim=16 +net_lr_bs256=5e-5 +bs=96 +epochs=800
...
wandb:
wandb: Run history:
wandb:            global_step ▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▄▅▅▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇██
wandb:               test/acc ▁▂▃▂▃▄▄▃▄▅▄▅▅▆▅▆▆▆▇▆▆▆▇▇▇▇▇█▇▇██▇█▇█████
wandb:              train/inv ██▇▇▆▄▄▄▃▄▃▃▃▃▂▃▂▃▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▁▂▂▁▂▂▂
wandb:      train/lejepa_loss █▄▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▁▂▁▂▁▁▂▁▁▁▂▂▂▂▁
wandb:               train/lr ██████▇▇▇▇▇▇▆▆▆▆▅▅▅▅▄▄▄▄▃▃▃▃▃▃▃▂▂▂▂▂▁▁▁▁
wandb: train/orig_lejepa_loss ███▇▇▆▅▅▄▄▄▃▃▃▃▃▃▂▃▃▂▃▂▃▂▂▂▂▂▂▁▁▂▂▁▂▂▁▁▂
wandb:       train/probe_loss █▃▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▁▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:           train/sigreg ▆▆█▆▅▆▇▃▃▇▃▅▄▅▄▅▄▅▄▂▃▃▄▅▃▃▃▂▃▃▁▄▃▄▁▂▂▂▅▁
wandb:           train/within ██▄▆▅▄▃▃▃▂▂▂▂▂▁▂▁▃▂▂▁▁▃▁▁▁▂▂▁▁▂▂▁▃▁▁▁▁▂▁
wandb:
wandb: Run summary:
wandb:            global_step 800
wandb:               test/acc 0.59057
wandb:              train/inv 0.3265
wandb:      train/lejepa_loss 0.01392
wandb:               train/lr 1e-05
wandb: train/orig_lejepa_loss 0.33261
wandb:       train/probe_loss 1.4414
wandb:           train/sigreg 0.63281
wandb:           train/within 0.00128
```


[^1]: For a video setup, I would use two spatial dimensions and 1 temporal dimension.
[^2]: With one unavoidable modification to the LeJepa baseline: in order to get the demo to utilize the final embedding layer of the ViT,
I had to change the embedding used for prediction from "CLS token" to "attention pooled across the final embedding layer".
