import os
import re
import json


def get_files(dir_name, ext='csv'):
    files = []
    for file in os.listdir(dir_name):
        if file.endswith(ext):
            files.append(os.path.join(dir_name, file))
    return files


def read_multiple_json_objects(document, pos=0, decoder=json.JSONDecoder()):

    not_whitespace = re.compile(r'[^\s]')

    while True:
        match = not_whitespace.search(document, pos)
        if not match:
            return
        pos = match.start()

        try:
            obj, pos = decoder.raw_decode(document, pos)
        except json.JSONDecodeError:
            # do something sensible if there's some error
            raise
        yield obj


def check_file_existence(filename):
    if not os.path.exists(filename):
        raise ValueError("Dataset file not found.")


def check_multiple_file_existence(filenames):
    for filename in filenames:
        check_file_existence(filename)


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

