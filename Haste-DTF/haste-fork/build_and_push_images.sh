#!/bin/bash

# Variables
ACR_NAME=<REPLACE_ME>
IMAGE_DIR=all
IMAGE_TAG=latest
REPO_DIR=$(pwd)
# Azure Batch nodes are GPU nodes, so the pushed training image keeps the CUDA
# wheels. This is also the training Dockerfile's own default, so the published
# image is unchanged by the local stack's CPU build. Override with `-v cpu` only
# when deliberately publishing an image for CPU-only Batch pools -- it drops
# ~3.2 GB (nvidia/* and triton) but will never use a GPU even if one is present.
TORCH_VARIANT=cu124

# Function to display usage
usage() {
    echo "Usage: $0 [-t image_tag] [-i image_dir] [-v torch_variant]"
    # corresponds to the docker directory name
    echo "Valid image directory names: training, imageryprep, all"
    echo '"all" will build both images'
    echo "Valid torch variants: cu124 (default, GPU Batch nodes), cpu"
    exit 1
}

# Parse named options
while getopts ":t:i:v:" opt; do
    case ${opt} in
        t )
            IMAGE_TAG=$OPTARG
            ;;
        i )
            IMAGE_DIR=$OPTARG
            ;;
        v )
            TORCH_VARIANT=$OPTARG
            ;;
        \? )
            usage
            ;;
    esac
done

# Validate TORCH_VARIANT: a typo here silently produces wheels pip cannot find
# and fails deep inside the conda solve with an unhelpful message.
case $TORCH_VARIANT in
    cu124|cpu)
        ;;
    *)
        echo "Error: Invalid torch variant '$TORCH_VARIANT'. Expected cu124 or cpu."
        usage
        ;;
esac

# Ensure IMAGE_NAME and IMAGE_TAG are set
if [ -z "$IMAGE_DIR" ] || [ -z "$IMAGE_TAG" ]; then
    usage
fi

# Validate IMAGE_NAME against enumerated values
case $IMAGE_DIR in
    training|imageryprep|all)
        ;;
    *)
        echo "Error: Invalid image name '$IMAGE_NAME'."
        usage
        ;;
esac

if [ "$IMAGE_DIR" == "all" ]; then
    IMAGES_TO_BUILD=("training" "imageryprep")
else
    IMAGES_TO_BUILD=("$IMAGE_DIR")
fi

echo "Building images: ${IMAGES_TO_BUILD[@]} with tag: $IMAGE_TAG"

# Login to Azure Container Registry
az acr login --name $ACR_NAME

for IMAGE_DIR in "${IMAGES_TO_BUILD[@]}"; do

    IMAGE_NAME="haste$IMAGE_DIR"
    echo "Building image: $IMAGE_NAME with tag: $IMAGE_TAG"

    # Only the training image consumes TORCH_VARIANT; passing an unused build
    # arg to imageryprep would just emit a warning.
    BUILD_ARGS=()
    if [ "$IMAGE_DIR" == "training" ]; then
        BUILD_ARGS+=(--build-arg "TORCH_VARIANT=$TORCH_VARIANT")
        echo "Torch variant: $TORCH_VARIANT"
    fi

    # Build and tag the Docker image
    echo "Command: docker build ${BUILD_ARGS[*]} -t $ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG -f $REPO_DIR/docker/$IMAGE_DIR/Dockerfile $REPO_DIR"
    docker build "${BUILD_ARGS[@]}" -t "$ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG" -f "$REPO_DIR/docker/$IMAGE_DIR/Dockerfile" "$REPO_DIR"
    if [ $? -ne 0 ]; then
        echo "Error building image $IMAGE_NAME:$IMAGE_TAG"
        exit 1
    fi
    echo "Image $IMAGE_NAME:$IMAGE_TAG built successfully."

    # Push the Docker image to ACR
    echo "Pushing image: $IMAGE_NAME with tag: $IMAGE_TAG to registry: $ACR_NAME.azurecr.io"
    docker push "$ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG"
    if [ $? -ne 0 ]; then
        echo "Error pushing image $IMAGE_NAME:$IMAGE_TAG"
        exit 1
    fi
    echo "Image $IMAGE_NAME:$IMAGE_TAG pushed to $ACR_NAME.azurecr.io successfully."
done
echo "Done"