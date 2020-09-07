#!/bin/bash
if [ ! -e "./docker/Dockerfile" ]
then
	    echo "File ./docker/Dockerfile dese not exist, make sure you invoke the script in the right directory"
	        exit -1
	fi
docker build -t tbd-transformer-pytorch -f docker/Dockerfile .
