#!/usr/bin/env bash
[ ! -d _venv ] && python3 -m venv _venv --system-site-packages # assumes amd-hipdf is installed here
source _venv/bin/activate
python3 -m pip install -r sphinx/requirements.txt
source build.sh
deactivate
