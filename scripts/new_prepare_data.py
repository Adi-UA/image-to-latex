import os
import subprocess
from pathlib import Path

from image_to_latex.data.utils import Tokenizer, get_all_formulas, get_split


PROJECT_DIRNAME = Path(__file__).resolve().parents[1]
DATA_DIRNAME = PROJECT_DIRNAME / "data"
VOCAB_FILE = PROJECT_DIRNAME / "image_to_latex" / "data" / "vocab.json"
CLEANED_FILE = "im2latex_formulas.norm.new.lst"


def main():
    # Run adi_prepare_data.py to download and process the latex data
    subprocess.run(["python", os.path.join(PROJECT_DIRNAME, "adi_prepare_data.py")], check=True)
    os.chdir(DATA_DIRNAME)

    # Build vocabulary
    print("Building vocabulary...")
    all_formulas = get_all_formulas(CLEANED_FILE)
    _, train_formulas = get_split(all_formulas, "im2latex_train_filter.lst")
    tokenizer = Tokenizer()
    tokenizer.train(train_formulas)
    tokenizer.save(VOCAB_FILE)


if __name__ == "__main__":
    main()
