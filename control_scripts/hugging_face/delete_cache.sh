#!bin/sh

set -e

echo 'Will delete ./.cache'

rm -rf ./.cache

echo './.cache deleted! Will re-create the folder'

mkdir ./.cache