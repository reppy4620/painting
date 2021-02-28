Painting
===

Painting is sketch-colorization project leveraging various GANs.

**Note that I've not got good result yet.**

This project is powered by following packages.

- pytorch-lightning(1.2.0) - [github](https://github.com/PyTorchLightning/pytorch-lightning)
- hydra(1.0.6) - [github](https://github.com/facebookresearch/hydra)
- wandb - [official](https://wandb.ai/site)

**hydra** is prerelease version which is installed by below command.

```bash
$ pip install hydra-core --upgrade --pre
```

## Methods

- **Baseline**  
Normal colorization model.
  

- **Lightweight**  
Using LightweightGAN's training procedure.  
Reference: [open-review](https://openreview.net/forum?id=1Fqg133qRaI)
  

## Train
Before run training script, you have to make "configs/env/defaults_env.yaml" like below.

```yaml:default_env.yaml
data_dir: path/to/<image-dir>
save_dir: path/to/<save-dir>
```

Run training script.
```
$ python train.py method=<*method-name*>

<*method-name*>: e.g. baseline, which corresponds to file name of "confgs/method/*.yaml"
```
