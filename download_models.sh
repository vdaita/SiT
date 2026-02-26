#!/bin/bash

mkdir -p models

curl -L "https://huggingface.co/nyu-visionx/SiT-collections/resolve/main/SiT-S-2-256.pt?download=true" -o models/S.pt
curl -L "https://huggingface.co/nyu-visionx/SiT-collections/resolve/main/SiT-B-2-256.pt?download=true" -o models/B.pt
curl -L "https://huggingface.co/nyu-visionx/SiT-collections/resolve/main/SiT-XL-2-256.pt?download=true" -o models/XL.pt