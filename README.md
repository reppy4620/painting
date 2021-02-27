Painting
===

Painting is sketch-colorization project leveraging various GANs.

**Note that all training have been continued to fail.**

This project is powered by following packages.

- pytorch-lightning - [github](https://github.com/PyTorchLightning/pytorch-lightning)
- hydra - [github](https://github.com/facebookresearch/hydra)
- wandb - [official](https://wandb.ai/site)


## Methods

- **Baseline**  
Normal colorization model.
  

- **Lightweight**  
Using LightweightGAN's training procedure.  
Reference: [open-review](https://openreview.net/forum?id=1Fqg133qRaI)
  

## Train
Before run following command, you have to make "configs/env/defaults_env.yaml".

```yaml:default_env.yaml
# @package _group_
data_dir: path/to/<image-dir>
save_dir: path/to/<save-dir>
```
When there isn't first line, hydra cannot interpret it as "env" group.

```
python train.py method=<*method-name*>

<*method-name*>: e.g. baseline, which corresponds to file name of "confgs/method/*.yaml"
```
