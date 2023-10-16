# Image to LaTeX

An application that maps an image of a LaTeX math equation to LaTeX code.

This is a fork that was updated to work with Python 3.10 since the original repo is no longer maintained and newer GPUs require newer Lightning versions. This also fixes [a longstanding issue on the original repo related to using an older Python version](https://github.com/kingyiusuen/image-to-latex/issues/20) by moving everything to a conda environment. It also trains the model on the [normalized Im2latex-100k dataset](https://im2markup.yuntiandeng.com/data/) that has been put through [some additional preprocessing](https://github.com/Adi-UA/image-to-latex/blob/main/scripts/download_and_extract_data.py#L68-L94).

I got rid of the api, streamlit and docker since I didn't need them.

## Results

- My Best Run: https://wandb.ai/adioss/image-to-latex/runs/yd1cadrl
- Run Path: `adioss/image-to-latex/yd1cadrl`
- Drive Link: https://drive.google.com/drive/folders/1mrAksby1ljpUo7kdXY3u7SO1AvEV_QzI?usp=sharing (contains the best model checkpoint. You can download it from here if you don't want to use W&B)

## How To Use

### Prerequisites

Miniconda: https://docs.conda.io/en/latest/miniconda.html

### Setup

Clone the repository to your computer and position your command line inside the repository folder:

```
git clone https://github.com/kingyiusuen/image-to-latex.git
cd image-to-latex
```

Assuming you have Miniconda installed, run the following command to install the required packages:

```
./scripts/install
```

The install script will create and install the required packages in a virtual environment named `img2latex-env` that uses Python 3.10.

But, bash scripts create a new shell, so you will need to run the following command to activate the environment again before running of the scripts:

```
conda activate img2latex-env
```

### Using and/or Testing

To use or test the model, you may use either my best run or your own run. Follow the instructions below to use my best run (replace my best run path with your own run path if you want to use your own run):

```bash
# Download the best model checkpoint from W&B
python3 scripts/download_checkpoint.py --run_path adioss/image-to-latex/yd1cadrl

# Run the model on an image
python3 scripts/test.py --image_path data/test/images/abcd.png
```

The results will be printed to the console.

### Training

#### Data Preparation

Run the following command to download and preprocess the data:

```

python3 scripts/prepare_data.py

```

This will:

- Download the data to `data`.
- Apply some preprocessing to normalize the LaTeX (in addition to the normalization applied by the original data owners)
- Apply some preprocessing to normalize the images (in addition to the normalization applied by the original data owners). This involves making them greyscale, padding them and skipping the ones with no text.

#### Actual Training

Run the following command to train the model:

```

python3 scripts/train.py

```

You can update the training configuration by editing `conf/config.yaml`. The default configuration is the one used to train the best model.

#### Experiment Tracking using Weights & Biases

The best model checkpoint will be uploaded to [Weights & Biases (W&B)](https://wandb.ai/) automatically (you will be asked to register or login to W&B before the training starts).

You can download the best model checkpoint from W&B using the following command:

```

python3 scripts/download_checkpoint.py RUN_PATH

```

Replace RUN_PATH with the path of your run (ex `adioss/image-to-latex/yd1cadrl`).

### Results

The results of our best model on the test set have been saved to the `results` directory. It was created with the following commands:

1. Activate the environment:

   ```bash
   conda activate img2latex-env
   ```

2. Download the best model checkpoint from W&B:

   ```bash
   python3 scripts/download_checkpoint.py # Default run path is adioss/image-to-latex/yd1cadrl
   ```

3. Run the evaluation script:
   ```bash
   python3 scripts/get_im2latex100k_test_results.py
   ```
