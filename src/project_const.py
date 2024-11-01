#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from pathlib import Path

PROJECT_ROOT_PATH = Path(__file__).absolute().parent.parent
DATA_PATH = PROJECT_ROOT_PATH / 'data'
SRC_PATH = PROJECT_ROOT_PATH / 'src'

API_URL = 'https://api-key.fusionbrain.ai/'

logging.basicConfig(format='%(asctime)s %(name)s:%(levelname)s:%(message)s',
                    level=logging.INFO)

if __name__ == '__main__':
    pass