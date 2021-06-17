#!/usr/bin/bash

docker run --rm -it --name container -p 8888:8888 -p 6006:6006 -p 5000:5000 -v "$(pwd)":/home/jovyan/work -v $HOME/ProjectsHDD/envs:/opt/conda/envs --gpus "all" --cap-add=SYS_ADMIN --ipc=host -u 1000:1000 ml start-notebook.sh $1
