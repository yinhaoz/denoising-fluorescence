#!/bin/bash

function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2 --no-check-certificate
  rm -rf /tmp/cookies.txt
}

GDRIVE_ID=1kHJUqb-e7BARb63741DVdpg-WqCdG3z6
TAR_FILE=./experiments/pretrained.tar

mkdir -p ./experiments

gdrive_download $GDRIVE_ID $TAR_FILE
tar -xvf $TAR_FILE -C ./experiments/
rm $TAR_FILE