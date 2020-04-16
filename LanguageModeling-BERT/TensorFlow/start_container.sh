docker build -t bert:v1 -
docker run --runtime=nvidia -it -v "$(pwd)":/bert bert:v1