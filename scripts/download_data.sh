#!/bin/bash
# Download datasets into the CoOp layout that trustprompt/datasets.py expects.
#   bash scripts/download_data.sh /data fast4      # DTD, EuroSAT, Flowers102, Caltech101 (go/no-go, ~1.2 GB)
#   bash scripts/download_data.sh /data rest       # Pets, Aircraft, Food101, SUN397 (+ manual: Cars, UCF101)
#   bash scripts/download_data.sh /data dg         # ImageNet-A, -R (+ manual: -V2, -Sketch)
#   bash scripts/download_data.sh /data all
# URLs were written from memory of CoOp's DATASETS.md and the official dataset pages, not tested.
# If one fails, open CoOp's DATASETS.md (github.com/KaiyangZhou/CoOp) and update it here.
# The split_zhou_*.json files are on Google Drive links in DATASETS.md: download them manually
# (or with gdown) and place each in its dataset folder; the script tells you where.
set -e
set -o pipefail
ROOT=${1:?usage: download_data.sh ROOT [fast4|rest|dg|all|dataset ...]}
shift; JOBS=${@:-fast4}
mkdir -p "$ROOT"; cd "$ROOT"
get() { [ -f "$2" ] && return 0; curl -4 -fL --connect-timeout 30 --retry 3 --retry-delay 10 -C - -o "$2.part" "$1" && mv "$2.part" "$2" || { rm -f "$2.part"; return 1; }; }
need_split() { [ -f "$1/$2" ] && echo "  ok: $1/$2" || echo "  MANUAL: put $2 (from CoOp DATASETS.md) into $ROOT/$1/"; }

dtd() {
  echo "== DTD"; mkdir -p tmp
  get https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz tmp/dtd.tar.gz
  [ -d dtd/images ] || tar -xzf tmp/dtd.tar.gz            # -> dtd/images/<class>/...
  need_split dtd split_zhou_DescribableTextures.json
}
eurosat() {
  echo "== EuroSAT"; mkdir -p tmp eurosat
  # primary DFKI link; fallback: torchgeo's copy on Hugging Face (URL from memory - verify)
  get https://madm.dfki.de/files/sentinel/EuroSAT.zip tmp/eurosat.zip || \
  get https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSAT.zip tmp/eurosat.zip
  [ -d eurosat/2750 ] || unzip -q tmp/eurosat.zip -d eurosat   # -> eurosat/2750/<class>/...
  need_split eurosat split_zhou_EuroSAT.json
}
flowers() {
  echo "== Flowers102"; mkdir -p tmp oxford_flowers
  get https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz tmp/flowers.tgz
  get https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat oxford_flowers/imagelabels.mat
  [ -d oxford_flowers/jpg ] || tar -xzf tmp/flowers.tgz -C oxford_flowers   # -> oxford_flowers/jpg/
  need_split oxford_flowers split_zhou_OxfordFlowers.json
}
caltech() {
  echo "== Caltech101"; mkdir -p tmp caltech-101
  # Original vision.caltech.edu link moved to data.caltech.edu; if this fails, download
  # caltech-101.zip from https://data.caltech.edu (search "Caltech 101") and extract 101_ObjectCategories.
  get https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip tmp/caltech101.zip
  if [ ! -d caltech-101/101_ObjectCategories ]; then
    unzip -q tmp/caltech101.zip -d tmp/caltech_x
    tar -xzf tmp/caltech_x/caltech-101/101_ObjectCategories.tar.gz -C caltech-101
  fi
  need_split caltech-101 split_zhou_Caltech101.json
}
pets() {
  echo "== OxfordPets"; mkdir -p tmp oxford_pets
  get https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz tmp/pets_img.tar.gz
  get https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz tmp/pets_ann.tar.gz
  [ -d oxford_pets/images ] || tar -xzf tmp/pets_img.tar.gz -C oxford_pets
  [ -d oxford_pets/annotations ] || tar -xzf tmp/pets_ann.tar.gz -C oxford_pets
  need_split oxford_pets split_zhou_OxfordPets.json
}
aircraft() {
  echo "== FGVCAircraft"; mkdir -p tmp
  get https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz tmp/fgvc.tar.gz
  if [ ! -d fgvc_aircraft ]; then tar -xzf tmp/fgvc.tar.gz -C tmp && mv tmp/fgvc-aircraft-2013b/data fgvc_aircraft; fi
  # needs fgvc_aircraft/{images, variants.txt, images_variant_train/val/test.txt}; no json split
}
food() {
  echo "== Food101 (~5 GB)"; mkdir -p tmp
  get http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz tmp/food101.tar.gz
  [ -d food-101/images ] || tar -xzf tmp/food101.tar.gz        # -> food-101/images/
  need_split food-101 split_zhou_Food101.json
}
sun() {
  echo "== SUN397 (~37 GB)"; mkdir -p tmp sun397
  get http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz tmp/sun397.tar.gz
  [ -d sun397/SUN397 ] || tar -xzf tmp/sun397.tar.gz -C sun397
  need_split sun397 split_zhou_SUN397.json
}
manual_rest() {
  echo "== MANUAL"
  echo "  StanfordCars: official link is down; use a Kaggle mirror, put images under $ROOT/stanford_cars/ to match split_zhou_StanfordCars.json"
  echo "  UCF101: download UCF-101-midframes from the Google Drive link in CoOp DATASETS.md -> $ROOT/ucf101/UCF-101-midframes"
  echo "  ImageNet: image-net.org (registration) or Kaggle ILSVRC -> $ROOT/imagenet/images/{train,val}/<wnid>/, plus classnames.txt from DATASETS.md"
}
dg() {
  echo "== ImageNet-A / -R"; mkdir -p tmp
  get https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar tmp/imagenet-a.tar
  get https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar tmp/imagenet-r.tar
  [ -d imagenet-adversarial ] || { mkdir -p imagenet-adversarial && tar -xf tmp/imagenet-a.tar -C imagenet-adversarial; }
  [ -d imagenet-rendition ] || { mkdir -p imagenet-rendition && tar -xf tmp/imagenet-r.tar -C imagenet-rendition; }
  echo "  MANUAL: ImageNet-V2 (matched-frequency) and ImageNet-Sketch: links in CoOp DATASETS.md"
}

for WHAT in $JOBS; do
  case $WHAT in
    fast4) dtd; eurosat; flowers; caltech ;;
    rest)  pets; aircraft; food; sun; manual_rest ;;
    dg)    dg ;;
    all)   dtd; eurosat; flowers; caltech; pets; aircraft; food; sun; manual_rest; dg ;;
    dtd|eurosat|flowers|caltech|pets|aircraft|food|sun) $WHAT ;;
    *) echo "unknown job $WHAT (fast4 rest dg all dtd eurosat flowers caltech pets aircraft food sun)"; exit 1 ;;
  esac
done
echo "done. Next: python scripts/check_data.py $ROOT"
