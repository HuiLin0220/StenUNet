#!/usr/bin/env bash

bash ./build.sh

docker save grand_challenge_algorithm | gzip -c > grand_challenge_algorithm.tar.gz
