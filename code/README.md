# Setup

```shell
conda create -n levyjepa python=3.10 -y
conda activate levyjepa
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install scipy tensorboard timm wandb hydra-core datasets==2.14.6 huggingface-hub "pyarrow<13.0.0" "numpy<2.0"
```

# Experiment 1

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
