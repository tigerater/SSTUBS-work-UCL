## Miner

This script will complete the given SStuBs with additional data about file length, file depth and commit delay.

## Prerequisites

- Docker

## How to run

Copy your SStuBs file into the `./src` folder and run the following commands:

    make
    make run

These will build and run a docker container that will pull all projects in your SStuBs file, gather data from the local files and output the resulst to `./src/enrichedYOUR_SSTUBS_FILE_NAME`

## Note

When running the script, all the projects in the sstubs file will be pulled one by one and it may take a long time.
Enriched results are saved after every SStuB is processed, but, rewritten when the script is started. Therefore, make sure you backup your `./src/enrichedYOUR_SSTUBS_FILE_NAME` file before restarting the script in case you need it.
