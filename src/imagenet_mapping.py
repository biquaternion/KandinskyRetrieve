#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import List, Dict

from src.project_const import DATA_PATH


class ImagenetMapping:
    def __init__(self):
        self.classes_path = DATA_PATH / 'imagenet_1000_classes.txt'
        self.classes = []
        self.get_classes()

    def get_classes(self) -> List[List[str]]:
        with open(self.classes_path) as f:
            for line in f:
                self.classes.append(line.strip().split(', '))
        return self.classes

    def save_as_json(self, output_path: Path = DATA_PATH / 'imagenet_1000.json') -> Dict[str, List[str]]:
        result = {str(i + 1): class_names for i, class_names in enumerate(self.classes)}
        with open(output_path, 'w') as f:
            f.write(json.dumps(result, indent=4))
        return result


if __name__ == '__main__':
    imapper = ImagenetMapping()
    classes = imapper.save_as_json()
    c = 0
    for k, v in classes.items():
         c += len(v)
    print(c)


