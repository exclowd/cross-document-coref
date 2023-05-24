
from curses.ascii import isspace
import glob
import json
import os
import subprocess
import xml.etree.ElementTree as ET

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

import csv

MAP = {}

def get_data_from_file(file):
    tree = ET.parse(file)
    file = os.path.basename(file)
    root = tree.getroot()
    leave = ['/', '']
    data = []
    for child in root.iter():
        if child.tag != 'token':
            continue
        tid = child.attrib.get('t_id')
        # print(MAP[file])
        if MAP.get(file, {}).get(tid, None) is not None:
            print(child.text.strip())
            data.append(f'<span class="entity{MAP[file][tid]}">'
                        f'{child.text.strip()}</span>')
        elif child.text is not None and child.text.strip() not in leave:
            data.append(child.text.strip())
    print(data)
    return ' '.join(data)

def get_data_from_topic(topic, topic_path):
    topic_data = {}
    for file_path in glob.glob(os.path.join(topic_path, '*.xml')):
        file = os.path.basename(file_path)
        data = get_data_from_file(file_path)
        topic_data[file] = data

    return topic_data



def get_data(data_dir):
    corpus_path = os.path.join(data_dir, 'ECB+')
    topic = '23'
    topic_dir = os.path.join(corpus_path, topic)
    if os.path.isdir(topic_dir):
        print(f"Processing {topic_dir}")
        data = get_data_from_topic(topic, topic_dir)
        # sort data by key
        data = dict(sorted(data.items(), key=lambda item: item[0]))
        with open( f"{topic}.txt", 'w+') as f:
            for doc, text in data.items():
                f.write(f'<h1>{doc}</h1>')
                f.write('<br><br>\n')
                f.write(text)
                f.write('<br><br>\n')
                f.write('<hr>\n')


def prepare_dataset(data_dir: str, output_dir: str) -> None:
    dataset_path = os.path.join(data_dir, 'ECB+_LREC2014')
    data = csv.reader(open('./results.csv', 'r'), delimiter='\t')
    global MAP
    curr = None
    for row in data:
        if row[-1].startswith('('):
            curr = row[-1].strip('()')
        if row[2] not in MAP:
            MAP[row[2]] = {}
        # print(row[2], row[4], curr)
        MAP[row[2]][row[4]] = curr
        if row[-1].endswith(')'):
            curr = None
    get_data(dataset_path)


def main():
    data_dir = "../data/ecbPlus"
    output_dir = "./out"
    prepare_dataset(data_dir, output_dir)


if __name__ == '__main__':
    main()
