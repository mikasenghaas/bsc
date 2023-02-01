#!/bin/bash

# install and setup pyenv if not installed
if ! command -v pyenv &> /dev/null
then
  # auto install from https://github.com/pyenv/pyenv
  curl https://pyenv.run | bash

  echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
  echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
  echo 'eval "$(pyenv init -)"' >> ~/.bashrc
  exit
fi

# setup virtualenv
pyenv virtualenv 3.10 bsc

# get requirements
pip install -r requirements.txt
