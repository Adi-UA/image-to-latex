# Image to LaTeX

An application that maps an image of a LaTeX math equation to LaTeX code. This is a fork that works exactly like the original, but is updated to support Python <=3.10 and the new setuptools rules. It also fixes a minor bug where Makefiles don't support source with `/bin/sh`, so I explicitly set its shell to bash. These changes fixe [a longstanding issue on the original repo](https://github.com/kingyiusuen/image-to-latex/issues/20).

Best run available here: https://wandb.ai/ternary-operators/image-to-latex/runs/i86j1ee0
val/cer: 0.071
val/loss: 0.13

## How To Use

### Setup

Clone the repository to your computer and position your command line inside the repository folder:

```
git clone https://github.com/kingyiusuen/image-to-latex.git
cd image-to-latex
```

Then, create a virtual environment named `venv` and install required packages:

```
make venv
```

This will create the virtual environment and install the required packages in it, but the nature of Makefiles means that we will exit the actual shell the makefile used to create the environment after we are done setting it up.

### Model Training and Experiment Tracking

#### Model Training

An example command to start a training session:

```
./train.sh
```

This will activate the venv and download and preprocess the data before training. The training session will be tracked by [Weights & Biases](https://wandb.ai/site). You will be asked to register or login to W&B before the training starts.

Configurations can be modified in `conf/config.yaml` or in command line. See [Hydra's documentation](https://hydra.cc/docs/intro) to learn more.

#### Experiment Tracking using Weights & Biases

The best model checkpoint will be uploaded to Weights & Biases (W&B) automatically (you will be asked to register or login to W&B before the training starts).

You can download the best model checkpoint from W&B using the following command:

```
python scripts/download_checkpoint.py RUN_PATH
```

Replace RUN_PATH with the path of your run (ex `ternary-operators/image-to-latex/i86j1ee0`). The run path for a particular experiment run, go to the Overview tab in the W&B dashboard.
