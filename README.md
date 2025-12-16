# LevyJepa
> Inspired by the [LeJepa minimal demo](https://github.com/rbalestr-lab/lejepa/blob/main/MINIMAL.md),
> I thought a Levy process would make a nice universal prior for video to incorporate the idea of "smooth spatialtemporal evolution punctuated by occassional discrete jumps".
>
> tl;dr: on the minimal demo, there is a miniscule but consistent lift, suggesting it might be useful for more complicated applications.

# Overview

* A latex document and PDF outlining the technique are in [the math subdirectory](math/).
* Code to run the experiments below is in [the code subdirectory](code/).

# Experiments

More details (including how to replicate) are in [the code subdirectory](code/README.md).

## Experiment 1: Add Levy Prior

Starting from the [LeJepa minimal demo](https://github.com/rbalestr-lab/lejepa/blob/main/MINIMAL.md), 
I added a 2D Levy prior[^1] to the final embedding layer of the ViT whose strength was controlled by parameter `gamma`:
`gamma=0` corresponds to the LeJepa baseline.[^2]

| What     | test/acc      | Notes  |
| ------------- |:-------------:| -----:|
| LeJepa baseline (no Levy 2D regularizer)  | 0.90726       | `gamma=0` |
| ibid with Levy 2D regularizer    | 0.91287       | `gamma=1` |

It's a tiny lift.  Honestly less exciting than I hoped.

## Experiment 2: Remove invariance loss

On the hunch that the [LeJepa minimal demo](https://github.com/rbalestr-lab/lejepa/blob/main/MINIMAL.md) is saturated, I created a parameter `kappa`
to control the strength of the invariance loss, with `kappa=0` corresponding to no invariance loss.  The settings in experiment 1 correspond to `kappa=1`.

| `gamma`       | test/acc      | Notes  |
| ------------- |:-------------:| -----:|
| LeJepa baseline without invariance loss (no Levy 2D regularizer) | 0.56943       |  `gamma=0`, `kappa=0` |
| ibid with Levy 2D regularizer  | 0.59057       | `gamma=1`, `kappa=0`     | 

That's a slightly larger lift.  This suggests to me the regularizer is helping.

# Thoughts

The lifts are tiny but consistent in my experiments, suggesting this is helpful, but not substantially.  
It could be that whole single image classification doesn't benefit that much a (sub)structural prior,
and a more complicated problem (image segmentation, video object tracking) might benefit more.  Or maybe I have a bug somewhere.

Overall this was a fun distraction and I'll look for the opportunity to exploit "distributional regularization" in other projects.

[^1]: For a video setup, I would use two spatial dimensions and 1 temporal dimension.
[^2]: With one unavoidable modification to the LeJepa baseline: in order to get the demo to utilize the final embedding layer of the ViT, 
I had to change the embedding used for prediction from "CLS token" to "attention pooled across the final embedding layer".

