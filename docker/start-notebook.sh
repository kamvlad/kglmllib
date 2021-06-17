#!/bin/bash
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

set -e

if [ ! -z "$1" ]
then
  if [ ! -d "/opt/conda/envs/$1" ]
  then
    conda create -y -n $1 --clone base
  fi
  
  if [ ! "$1" = "base" ]
  then
    . /opt/conda/etc/profile.d/conda.sh && conda activate $1 && jupyter lab --no-browser --ip=0.0.0.0
  fi
else
  jupyter lab --no-browser --ip=0.0.0.0
fi


