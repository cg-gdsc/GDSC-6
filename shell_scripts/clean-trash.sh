#!/bin/bash
# This script installs a single pip package on a SageMaker Studio Kernel Application

set -eux

rm -r .local/share/Trash/info || true
rm -r .local/share/Trash/files || true