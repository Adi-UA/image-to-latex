# Image to LaTeX

An application that maps an image of a LaTeX math equation to LaTeX code. This is a fork that works exactly like the original, but is updated to support Python <=3.10 and the new setuptools rules. It also fixes a minor bug where Makefiles don't support source with `/bin/sh`, so I explicitly set its shell to bash. These changes fixe [a longstanding issue on the original repo](https://github.com/kingyiusuen/image-to-latex/issues/20).

I will a link to my own best run for this model in the future along with the BLEU score metric.

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

This will create the virtual environment and install the required packages in it, but the nature of Makefiles means that we will exit the actual shell the makefile used to create the environment after we are done setting it up. So, to actually run the scripts below, you will need to activate the venv again in your terminal with:

```
source ./venv/bin/activate
```

This alreadyhas the modules required as a result of the previous step. The steps after this are exactly the same as the original repository.

### Data Preprocessing

Run the following command to download the im2latex-100k dataset and do all the preprocessing. (The image cropping step may take over an hour.)

```
python scripts/prepare_data.py
```

### Model Training and Experiment Tracking

#### Model Training

An example command to start a training session:

```
python scripts/run_experiment.py
```

Configurations can be modified in `conf/config.yaml` or in command line. See [Hydra's documentation](https://hydra.cc/docs/intro) to learn more.

#### Experiment Tracking using Weights & Biases

The best model checkpoint will be uploaded to Weights & Biases (W&B) automatically (you will be asked to register or login to W&B before the training starts). Here is an example command to download a trained model checkpoint from W&B:

```
python scripts/download_checkpoint.py RUN_PATH
```

Replace RUN_PATH with the path of your run. The run path should be in the format of `<entity>/<project>/<run_id>`. To find the run path for a particular experiment run, go to the Overview tab in the dashboard.

The checkpoint will be downloaded to a folder named `artifacts` under the project directory.
