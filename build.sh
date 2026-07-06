#!/usr/bin/env bash
set -euo pipefail
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# Build for the challenge's linux/amd64 evaluation environment.
docker build --platform linux/amd64 -t puma-challenge-baseline-track2 "$SCRIPTPATH"
