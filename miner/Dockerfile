FROM ubuntu:20.04

WORKDIR /home

# Install misc tools
RUN apt update
RUN apt -y install curl
RUN apt -y install git

# Install Node.js 12
RUN curl https://deb.nodesource.com/setup_12.x | bash
RUN apt -y install nodejs
