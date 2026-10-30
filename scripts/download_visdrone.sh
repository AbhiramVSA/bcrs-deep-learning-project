#!/usr/bin/env bash
# Download VisDrone-DET 2019 (train/val/test-dev) into data/visdrone/.
# Works on the A100 server and on Windows git-bash.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/visdrone
cd data/visdrone

BASE="https://github.com/ultralytics/assets/releases/download/v0.0.0"
for split in train val test-dev; do
  zip="VisDrone2019-DET-${split}.zip"
  if [ -d "VisDrone2019-DET-${split}" ]; then
    echo "already extracted: ${split}"
    continue
  fi
  if [ ! -f "$zip" ]; then
    echo "downloading ${zip} ..."
    # -f: fail on HTTP errors; -C -: resume a partial .part across reruns;
    # mv only on success so a truncated file never masquerades as complete
    curl -fL --retry 3 -C - -o "${zip}.part" "${BASE}/${zip}"
    mv "${zip}.part" "$zip"
  fi
  echo "extracting ${zip} ..."
  if command -v unzip >/dev/null 2>&1; then
    unzip -q "$zip"
  else
    python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall()" "$zip"
  fi
  rm -f "$zip"
done

echo "--- verification ---"
for split in train val test-dev; do
  n_img=$(ls "VisDrone2019-DET-${split}/images" | wc -l)
  n_ann=$(ls "VisDrone2019-DET-${split}/annotations" | wc -l)
  echo "${split}: ${n_img} images, ${n_ann} annotation files"
done
echo "expected: train 6471, val 548, test-dev 1610"
