# Сборка образа

docker build -t ml .

# Запуск

docker run --rm -it --name container -p 8888:8888 -p 6006:6006 -p 5000:5000 -v "$(pwd)":/home/jovyan/work -v $HOME/ProjectsHDD/envs:/opt/conda/envs --gpus "all" --cap-add=SYS_ADMIN --ipc=host -u 1000:1000 ml start-notebook.sh [имя среды]

Или

./notebook.sh [имя среды]

Лучше сделать alias в .bashrc

# Порты

8888 - jupyter lab
6006 - tensorboard
5000 - mlflow

# Пути

$HOME/ProjectsHDD/envs - путь к conda средам
