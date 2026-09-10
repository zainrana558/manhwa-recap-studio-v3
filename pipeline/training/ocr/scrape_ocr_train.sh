#!/usr/bin/env bash
# scrape 10 chapters each of hard manhwa/webtoons for OCR training data
export PATH="$HOME/.bun/bin:$PATH"
cd /home/azureuser/manhwa-recap-studio/mini-services/pipeline-service
OUT=/home/azureuser/manhwa-recap-studio/data/ocr_train_raw
mkdir -p "$OUT"
scrape() {  # source id slug count
  echo "=== $1 / $2 (x$4) ==="
  timeout 900 bun scrape_chapters.ts --source "$1" --id "$2" --count "$4" --out "$OUT/$3" 2>&1 | tail -4
}
scrape asurascans return-of-the-mount-hua-sect        mounthua      10
scrape mgeko      northern-b-manhwa                    northernblade 10
scrape asurascans omniscient-readers-viewpoint         orv           10
scrape asurascans sss-class-suicide-hunter             sssclass      10
scrape asurascans damn-reincarnation                   damnreinc     10
scrape asurascans reaper-of-the-drifting-moon          reaper        10
scrape asurascans the-great-mage-returns-after-4000-years greatmage   10
scrape mgeko      the-breaker22                         breaker       10
scrape mgeko      the-beginning-after-the-end           tbate         10
scrape webtoons   1049                                  lookism       10
scrape webtoons   1571                                  eleceed       10
echo "ALL SCRAPES DONE"
du -sh "$OUT"/* 2>/dev/null
