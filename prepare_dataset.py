import argparse
import glob
import json
import subprocess
import os
import xml.etree.ElementTree as ET

from tqdm import tqdm

DATA_DIR = "./data/ecbPlus"
OUTPUT_DIR = "./data/ecbPlusOut"


def get_data_from_file(topic, file):
    tree = ET.parse(file)
    root = tree.getroot()

    event_mentions = []
    entity_mentions = []

    sentence_dict = {}
    vocab = set()

    token_tag_dict = {}
    mentions_dict = {}
    relations_dict = {}

    for child in root:
        n = child.attrib.get('sentence')
        if n is None:
            continue

        cur_sentence = sentence_dict.get(n, [])
        cur_sentence.append([child.attrib.get('t_id'), child.text])
        token_tag_dict[child.attrib.get('t_id')] = child
        sentence_dict[n] = cur_sentence
        vocab.update([child.text])

    for mention in root.find('Markables'):
        if mention.attrib.get('RELATED_TO') is None:
            t_ids = []
            for term in mention:
                t_ids.append(term.attrib.get('t_id'))

            if mention.tag.startswith('ACTION') or mention.tag.startswith('NEG_ACTION'):
                mention_type = 'event'
            else:
                mention_type = 'entity'

            sentence = token_tag_dict[t_ids[0]].attrib.get('sentence')
            term = [token_tag_dict[t_id].text for t_id in t_ids]

            sentence_words = [word for _, word in sentence_dict[sentence]]

            left = [word for t_id, word in sentence_dict[sentence] if t_id < t_ids[0]]
            right = [word for t_id, word in sentence_dict[sentence] if t_id > t_ids[-1]]

            m = {
                'doc_id': root.attrib.get('doc_id'),
                'topic': topic,
                'type': mention_type,
                'sent_id': sentence,
                'm_id': mention.attrib.get('m_id'),
                'mention_type': mention.tag,
                't_ids': t_ids,
                'term': term,
                'sentence': sentence_words,
                'left': left,
                'right': right
            }
            mentions_dict[m['m_id']] = m
        else:
            m_id = mention.attrib.get('m_id')
            relations_dict[m_id] = {
                'coref_chain': mention.attrib.get('instance_id', ''),
                'cluster_desc': mention.attrib['TAG_DESCRIPTOR']
            }

    relation_source_target = {}
    relation_rid = {}
    relation_tag = {}

    for relation in root.find('Relations'):
        target_mention = relation[-1].attrib['m_id']
        relation_tag[target_mention] = relation.tag
        relation_rid[target_mention] = relation.attrib['r_id']
        for mention in relation:
            if mention.tag == 'source':
                relation_source_target[mention.attrib['m_id']] = target_mention

    for mention, dic in mentions_dict.items():
        target = relation_source_target.get(mention, None)
        desc_cluster = ''
        if target is None:
            id_cluster = 'Singleton_' + dic['mention_type'] + '_' + dic['m_id'] + '_' + dic['doc_id']
        else:
            r_id = relation_rid[target]
            tag = relation_tag[target]

            if tag.startswith('INTRA'):
                id_cluster = 'INTRA_' + r_id + '_' + dic['doc_id']
            else:
                id_cluster = relations_dict[target]['coref_chain']

            desc_cluster = relations_dict[target]['cluster_desc']

        mention_obj = dic.copy()
        mention_obj['coref_chain'] = id_cluster
        mention_obj['cluster_desc'] = desc_cluster

        if mention_obj['type'] == 'event':
            event_mentions.append(mention_obj)
        else:
            entity_mentions.append(mention_obj)

    return event_mentions, entity_mentions, list(sentence_dict.values()), vocab


def get_data_from_dir(path):
    event_mentions = []
    entity_mentions = []
    sentences = []
    vocab = set()

    for file in glob.glob(os.path.join(path, '*.xml')):
        event, ent, sent, voc = get_data_from_file(os.path.basename(path), file)
        event_mentions.extend(event)
        entity_mentions.extend(ent)
        sentences.extend(sent)
        vocab.update(voc)

    return event_mentions, entity_mentions, sentences, vocab


def get_data(dataset_path, output_dir):
    events = []
    entities = []
    sentences = []
    vocab = set()

    corpus_path = os.path.join(dataset_path, 'ECB+')

    for dir in tqdm(os.listdir(corpus_path)):
        path = os.path.join(corpus_path, dir)
        if os.path.isdir(path):
            event, ent, sent, voc = get_data_from_dir(path)
            events.extend(event)
            entities.extend(ent)
            sentences.extend(sent)

            vocab.update(voc)

    return events, entities, sentences, vocab


def prepare_dataset(args):
    data_dir = args.data_dir
    output_dir = args.output_dir
    # check if data directory exists
    if not (os.path.exists(data_dir) and os.path.isdir(data_dir)):
        raise Exception("Data directory does not exist")

    # make output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_path = os.path.join(data_dir, 'ECB+_LREC2014')
    zip_path = os.path.join(DATA_DIR, 'ECB+_LREC2014', 'ECB+.zip')

    if not os.path.exists(zip_path):
        raise Exception("ECB+ dataset not found in data directory")

    try:
        subprocess.run(['unzip', '-o', zip_path, '-d', os.path.join(DATA_DIR, 'ECB+_LREC2014')])
    except subprocess.CalledProcessError as e:
        raise Exception("Error while unzipping ECB+ dataset")

    events, entities, sentences, vocab = get_data(dataset_path, output_dir)

    with open(os.path.join(output_dir, 'event_gold_mentions.json'), 'w+') as f:
        json.dump(events, f)

    with open(os.path.join(output_dir, 'entity_gold_mentions.json'), 'w+') as f:
        json.dump(entities, f)

    with open(os.path.join(output_dir, 'sentences.json'), 'w+') as f:
        json.dump(sentences, f)

    with open(os.path.join(output_dir, 'vocab.txt'), 'w') as f:
        for word in vocab:
            f.write(word + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='prepare_dataset.py',
        description='Prepare ECB+ dataset for training and testing',
        epilog='Example: python prepare_dataset.py --data-dir ./data/ecbPlus --output-dir ./data/ecbPlusOut'
    )
    parser.add_argument('--data-dir', type=str, default=DATA_DIR, help='Path to ECB+ dataset')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR, help='Path to output directory')
    arguments = parser.parse_args()
    prepare_dataset(arguments)
