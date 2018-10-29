#!/usr/bin/env bash

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    # for pysoundfile in ubuntu based operating systems
    sudo apt install libsndfile1
elif [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ "$(which brew)" != "brew not found" ]]; then
        # for librosa in macos
        brew install ffmpeg
    else
        echo "Please install brew package manager to install fmpeg for librosa"
    fi
fi