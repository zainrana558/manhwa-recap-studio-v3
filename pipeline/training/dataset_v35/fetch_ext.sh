#!/usr/bin/env bash
# Re-fetch the external annotation sets used by assemble_v35.py.
# (gitignored — ~23 MB, re-clonable, licences included in each repo)
set -e
cd "$(dirname "$0")"
mkdir -p ext_annotations && cd ext_annotations

# Manga109 public annotations: COO onomatopoeia (61k SFX polygons, CC-BY-4.0)
#                              + Manga109Dialog + MangaUB
if [ ! -d m109_public ]; then
  git clone --depth 1 https://github.com/manga109/public-annotations.git m109_public
  rm -rf m109_public/.git
fi

# CoMix / Comics Datasets Framework (DCM / eBDtheque / PopManga adapters + docs)
if [ ! -d cdf ]; then
  git clone --depth 1 https://github.com/emanuelevivoli/CoMix.git cdf
  rm -rf cdf/.git cdf/docs
fi

echo "ext_annotations ready:"
du -sh m109_public cdf 2>/dev/null || true
