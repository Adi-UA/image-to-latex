import json
import multiprocessing
import os
import sys

from scripts.test import test_img

project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(project_dir, "data")

with open(os.path.join(data_dir, "im2latex_test_filter.lst"), "r") as f:
    test_entries = f.readlines()
    test_data = [(entry.split()[0], int(entry.split()[1])) for entry in test_entries]

with open(os.path.join(data_dir, "im2latex_formulas.norm.new.lst"), "r") as f:
    formulae = f.readlines()


def collect_predictions(pid: int, test_data: list):
    results = {}
    errors = {}
    for img_name, line_no in test_data:
        print(f"Processing image {img_name}...")
        img_path = os.path.join(data_dir, "formula_images", img_name)
        if os.path.exists(img_path):
            try:
                ground_truth = formulae[line_no]
                prediction = test_img(img_path)
                results[img_name.rstrip(".png")] = (ground_truth, prediction)
            except Exception as e:
                errors[img_name.rstrip(".png")] = e
        else:
            print(f"Image {img_path} does not exist. Skipping.")

    return results, errors


if __name__ == "__main__":
    # Split test data into n chunks and run each chunk in parallel with Pool
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    chunk_size = len(test_data) // n
    chunks = [
        test_data[i : i + chunk_size] for i in range(0, len(test_data), chunk_size)
    ]
    with multiprocessing.Pool(n) as pool:
        results = pool.starmap(collect_predictions, enumerate(chunks))

    # Merge results
    all_results = {}
    all_errors = {}
    for result, error in results:
        all_results.update(result)
        all_errors.update(error)

    # Create results directory if it doesn't exist
    if not os.path.exists(os.path.join(project_dir, "results")):
        os.mkdir(os.path.join(project_dir, "results"))

    results_dir = os.path.join(project_dir, "results")

    # Save results
    with open(os.path.join(results_dir, "im2latex100k_test_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    with open(os.path.join(results_dir, "im2latex100k_test_errors.json"), "w") as f:
        json.dump(all_errors, f, indent=4)
