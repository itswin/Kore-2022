#!/usr/bin/env bash

cd src
cd KoreBeta
tar -czvf ../submission.tar.gz *.py

# kaggle competitions submit -c kore-2022 -f submission.py -m "Message"