#!/bin/bash
set -e

for course in */; do
  if [ -f "${course}videos.txt" ]; then
    mkdir -p "${course}videos"
    yt-dlp -o "%(id)s.%(ext)s" -S "res:720" -a "${course}videos.txt" -P "${course}videos" >> ./download_log.txt 2>&1
  fi
done
