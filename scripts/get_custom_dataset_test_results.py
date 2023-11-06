import argparse
import json
import multiprocessing
import os
import time
from typing import Tuple

import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from tqdm import tqdm

from image_to_latex.lit_models import LitResNetTransformer

project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def save_results(dataset: str, results: dict, errors: dict):
    if not os.path.exists(os.path.join(project_dir, "results")):
        os.mkdir(os.path.join(project_dir, "results"))

    results_dir = os.path.join(project_dir, "results")

    # Save results
    with open(os.path.join(results_dir, f"{dataset}_test_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    with open(os.path.join(results_dir, f"{dataset}_test_errors.json"), "w") as f:
        json.dump(errors, f, indent=4)

    print(f"Saved results to {results_dir}.")


def get_test_formulae(dataset: str) -> list:
    data_dir = os.path.join(project_dir, dataset)
    with open(os.path.join(data_dir, "latex.norm.lst"), "r") as f:
        formulae = [line.rstrip("\n") for line in f.readlines()]

    return formulae


def get_test_data_chunks(num_formulae: int, chunk_size: int) -> list:
    for i in range(0, num_formulae, chunk_size):
        yield list(range(i, min(i + chunk_size, num_formulae)))


def load_completed_result_ids(dataset: str) -> Tuple[dict, set]:
    results_dir = os.path.join(project_dir, "results")
    try:
        with open(os.path.join(results_dir, f"{dataset}_test_results.json"), "r") as f:
            results = json.load(f)
        ids = {int(k) for k in results.keys()}
        return results, ids
    except FileNotFoundError:
        return {}, set()


def predict(
    img_path: str,
    lit_model: LitResNetTransformer,
    transform: ToTensorV2,
) -> str:
    try:
        print(f"Processing {img_path}...")
        image = Image.open(img_path).convert("L")
        image_tensor = transform(image=np.array(image))["image"]

        pred = lit_model.model.predict(image_tensor.unsqueeze(0).float())[0]
        decoded = lit_model.tokenizer.decode(pred.tolist())
        decoded_str = " ".join(decoded)
        return decoded_str
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of processes to use. Defaults to 1.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="physics-10k",
        help="Dataset to use. Defaults to physics-10k.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20,
        help="Number of images to process per process. Defaults to 100.",
    )

    args = parser.parse_args()

    # Load model
    lit_model = LitResNetTransformer.load_from_checkpoint("artifacts/model.ckpt")
    lit_model.freeze()

    # Load transform
    transform = ToTensorV2()

    start_time = time.perf_counter()

    img_dir = os.path.join(project_dir, args.dataset, "images_processed")
    formulae = get_test_formulae(args.dataset)
    test_data_chunks = get_test_data_chunks(len(formulae), args.chunk_size)

    final_results = {}
    final_errors = []
    results, completed = load_completed_result_ids(args.dataset)
    final_results.update(results)
    for i, chunk in enumerate(test_data_chunks):
        print(f"Processing chunk {i}...")
        arg_batches = [
            (os.path.join(img_dir, f"{i}.png"), lit_model, transform)
            for i in chunk
            if i not in completed
        ]
        with multiprocessing.Pool(args.n) as pool:
            results = pool.starmap(
                predict,
                arg_batches,
            )

        for j, result in enumerate(results):
            if result is not None:
                final_results[chunk[j]] = {
                    "ground_truth": formulae[chunk[j]],
                    "prediction": result,
                }
            else:
                final_errors.append(chunk[j])

        completed.update(chunk)

        print(f"Finished processing chunk {i}.")
        print(f"Total errors: {len(final_errors)}")
        save_results(args.dataset, final_results, final_errors)

    # sort results by key
    final_results = dict(sorted(final_results.items(), key=lambda x: int(x[0])))
    save_results(args.dataset, final_results, final_errors)
    end_time = time.perf_counter()
    print(f"Finished in {end_time - start_time} seconds.")
