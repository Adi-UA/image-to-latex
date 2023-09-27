import os
import subprocess
from pathlib import Path

import image_to_latex.data.utils as utils
from image_to_latex.data.utils import Tokenizer, get_all_formulas, get_split


PROJECT_DIRNAME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIRNAME = PROJECT_DIRNAME / "data"
RAW_IMAGES_DIRNAME = DATA_DIRNAME / "formula_images"
PROCESSED_IMAGES_DIRNAME = DATA_DIRNAME / "formula_images_processed"
VOCAB_FILE = PROJECT_DIRNAME / "image_to_latex" / "data" / "vocab.json"
CLEANED_FILE = "im2latex_formulas.norm.new.lst"


def main():
    # Run adi_prepare_data.py to download and process the latex data
    subprocess.run(["python3", os.path.join(PROJECT_DIRNAME, "scripts" ,"download_and_extract_data.py")], check=True)
    os.chdir(DATA_DIRNAME)

    # Extract regions of interest
    if not PROCESSED_IMAGES_DIRNAME.exists():
        PROCESSED_IMAGES_DIRNAME.mkdir(parents=True, exist_ok=True)
        print("Cropping images...")
        for image_filename in RAW_IMAGES_DIRNAME.glob("*.png"):
            cropped_image = utils.crop(image_filename, padding=8)
            if not cropped_image:
                continue
            cropped_image.save(PROCESSED_IMAGES_DIRNAME / image_filename.name)

    # Build vocabulary
    print("Building vocabulary...")
    all_formulas = get_all_formulas(CLEANED_FILE)
    _, train_formulas = get_split(all_formulas, "im2latex_train_filter.lst")
    tokenizer = Tokenizer()
    tokenizer.train(train_formulas)
    tokenizer.save(VOCAB_FILE)


if __name__ == "__main__":
    main()
