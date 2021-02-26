Painting
===

Painting is sketch-colorization project leveraging various GANs.

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

```
python train.py method=<method-name>
```

"<method-name>" is selected by file name in "configs/method/*.yaml".
