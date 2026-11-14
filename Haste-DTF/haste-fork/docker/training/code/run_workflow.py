# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Run the segmentation-model inference workflow (inference, merge, visualizer).

Model training (create_masks.py + fine_tune.py) was removed: the image only
runs models that were trained elsewhere.
"""

import argparse
import glob
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone

import yaml


def run_subprocess(command, step_name):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        log_progress(f"Error running {step_name}")
        print(f"Error running {step_name} (exit code: {result.returncode})")
        if result.stdout:
            print(f"[{step_name}] stdout:\n{result.stdout}")
        if result.stderr:
            print(f"[{step_name}] stderr:\n{result.stderr}")
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config file"
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["inference"],
        default="inference",
        help=(
            "Step to execute. Only inference remains: model training was "
            "removed from this image (models arrive pre-trained)."
        ),
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.step == "inference":
        log_progress("Generating predictions")
        run_subprocess(
            ["python", "inference.py", "--config", args.config, "--overwrite"],
            "inference.py",
        )

        inference_dir = os.path.join(
            config.get("experiment_dir"),
            config.get("inference", {}).get("output_subdir"),
        )

        # Building footprints are downloaded once per image layer during the
        # imageryprep workflow and shipped here as inputs/building_footprints.gpkg
        # by the LocalRunner / Azure Batch runner. The previous in-workflow
        # extract_data_mask_from_geotiff.py + download_building_footprints.py
        # steps were removed when imageryprep started caching them.
        log_progress("Using cached building footprints")
        downloaded_footprints_fn = os.path.join(
            os.environ.get("AZ_BATCH_TASK_WORKING_DIR", "."),
            "inputs",
            "building_footprints.gpkg",
        )
        if not os.path.exists(downloaded_footprints_fn):
            log_progress(
                "Cached building footprints not found at "
                f"{downloaded_footprints_fn}; the image layer must be "
                "re-processed through imageryprep to produce them."
            )
            raise RuntimeError(
                f"Cached building footprints missing at {downloaded_footprints_fn}"
            )

        # Merge with inferred damage layer
        # Assumes there's only one *_predictions.tif

        log_progress("Merging predictions")

        predictions_files = glob.glob(
            os.path.join(inference_dir, "*_predictions.tif")
        )
        if not predictions_files:
            log_progress("Error running inference: no predictions files found")
            raise RuntimeError(
                "Something went wrong, no predictions files found in "
                f"{inference_dir}. Please download all artifacts for this model"
                " and check the stderr and stdout files for more information."
            )
        predicted_damage_fn = predictions_files[0]
        gpkg_prefix = config["inference"]["predictions_gpkg_fileprefix"]
        merged_building_predictions_fn = os.path.join(
            inference_dir, f"{gpkg_prefix}.gpkg"
        )
        run_subprocess(
            [
                "python",
                "merge_with_building_footprints.py",
                "--footprints_fn",
                downloaded_footprints_fn,
                "--predictions_fn",
                predicted_damage_fn,
                "--output_fn",
                merged_building_predictions_fn,
                "--overwrite",
            ],
            "merge_with_building_footprints.py",
        )

        # Generate visualizer output - same file name, but with a .tif extension
        log_progress("Generating results")
        temp_vis_fn = predicted_damage_fn.replace(
            "_predictions.tif", "_temp_vis.tif"
        )
        run_subprocess(
            [
                "python",
                "output2visualizer.py",
                "--merged_footprints_fn",
                merged_building_predictions_fn,
                "--predictions_fn",
                predicted_damage_fn,
                "--output_fn",
                temp_vis_fn,
                "--overwrite",
            ],
            "output2visualizer.py",
        )

        visualizer_cog_fn = temp_vis_fn.replace(
            "_temp_vis.tif", "_visualizer.tif"
        )

        log_progress("Converting results to COG")

        gdal_translate_params = os.getenv("GDAL_TRANSLATE_PARAMS")
        formatted_gdal_params = []
        if gdal_translate_params:
            if "-co" in gdal_translate_params:
                formatted_gdal_params = gdal_translate_params.split()
            else:
                for op in gdal_translate_params.split():
                    formatted_gdal_params += ["-co", op]
        run_subprocess(
            [
                "gdal_translate",
                "-of",
                "COG",
                *formatted_gdal_params,
                temp_vis_fn,
                visualizer_cog_fn,
            ],
            "gdal_translate",
        )


def log_progress(message):
    """Write progress steps and messages to a file"""
    # Create a directory for logs if it doesn't exist
    log_dir = os.path.join(os.getenv("AZ_BATCH_TASK_WORKING_DIR", "."), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "workflow_progress.log")

    with open(log_file, "a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()}|{message}\n")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(e)
        traceback.print_exc()
        log_progress(
            f"Error running command (exit code {e.returncode}). See task stderr for details."
        )
        log_progress("Please contact your HASTE website administrator")
        sys.exit(1)
    except Exception as e:
        print(e)
        traceback.print_exc()
        log_progress(
            f"Unexpected error: {type(e).__name__}. See task stderr for details."
        )
        log_progress("Please contact your HASTE website administrator")
        sys.exit(1)
