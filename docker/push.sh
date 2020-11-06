#!/bin/bash
# Pushes the docker image to a remote online repository, e.g., gcr.io
#
# Example:
# export IMAGE_URI=gcr.io/earthsea-dev/valan
# sh docker/push.sh

set -e
set -o pipefail


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
if [[ -n "$1" ]];
then
  LABEL="$1"
else
  LABEL="latest"
fi

# Example IMAGE_URI=gcr.io/$PROJECT_ID/valan
docker tag valan "${IMAGE_URI}":"${LABEL}"
docker push "${IMAGE_URI}":"${LABEL}"
