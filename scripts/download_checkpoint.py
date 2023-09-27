import argparse
import wandb
import os

def download_checkpoint(run_path: str) -> None:
    """
    Download best model checkpoint from Weights & Biases.
    """

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(project_dir, "artifacts")

    api = wandb.Api() 
    run = api.run(run_path)
    for artifact in run.logged_artifacts():
        if artifact.type == "model" and "best" in artifact.aliases:
            artifact.download(root=artifacts_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", type=str)
    args = parser.parse_args()
    download_checkpoint(args.run_path)


if __name__ == "__main__":
    main()
