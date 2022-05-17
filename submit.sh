#!/usr/bin/env bash

if [ ! $# -eq 2 ]; then
    echo "Usage: $0 <agent> <submit_message>"
    exit 1
fi

cd src
cd $1

TEST_FILE=main.py
if test -f "$TEST_FILE"; then
    tar -czvf submission.tar.gz *.py
    kaggle competitions submit -c kore-2022 -f submission.tar.gz -m "$2"
    rm submission.tar.gz
else
    echo "Could not find main.py. Did you enter the agent name correctly?"
    exit 1
fi
