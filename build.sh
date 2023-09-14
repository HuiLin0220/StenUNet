#!/usr/bin/env bash
SCRIPTPATH="$( cd "$('/home/hln0895/angiogram/phase_2/' "$0")" ; pwd -P )"

docker build -t grand_challenge_algorithm "$SCRIPTPATH"
