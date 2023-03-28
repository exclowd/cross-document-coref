import glob
import json
import os
import subprocess
import xml.etree.ElementTree as ET

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def get_data_from_file(file, sentences):
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


def get_clusters(mentions):
    clusters = {}
    for i, mention in enumerate(mentions):
        cluster_id = mention['coref_chain']
        clusters[cluster_id] = clusters.get(cluster_id, list).append(i)

    return clusters


def get_data_from_topic(topic, topic_path, validated_sentences):
    event_mentions = []
    entity_mentions = []

    for file_path in glob.glob(os.path.join(topic_path, '*.xml')):
        file = os.path.basename(file_path)
        if file in validated_sentences:
            selected_sentences = list(map(int, validated_sentences[file]))
            print(file, validated_sentences[file])
            event_mentions, entity_mentions = get_data_from_file(file_path, selected_sentences)
            event_mentions.extend(event_mentions)
            entity_mentions.extend(entity_mentions)

    event_clusters = get_clusters(event_mentions)
    entity_clusters = get_clusters(entity_mentions)

    event_singleton_cluster_flag = {c: True if len(m) == 1 else False for c, m in event_clusters.items()}
    entity_singleton_cluster_flag = {c: True if len(m) == 1 else False for c, m in entity_clusters.items()}

    for item in event_mentions:
        item.update({'topic': topic, 'singleton': event_singleton_cluster_flag[item['cluster_id']]})
    for item in entity_mentions:
        item.update({'topic': topic, 'singleton': entity_singleton_cluster_flag[item['cluster_id']]})

    return event_mentions, entity_mentions


def get_data(data_dir, validated_sentences):
    events = []
    entities = []

    corpus_path = os.path.join(data_dir, 'ECB+')

    for topic in os.listdir(corpus_path):
        topic_dir = os.path.join(corpus_path, topic)
        if os.path.isdir(topic_dir):
            print(f"Processing {topic_dir}")
            event_mentions, entity_mentions = get_data_from_topic(topic, topic_dir, validated_sentences[topic])
            events.extend(event_mentions)
            entities.extend(entity_mentions)


def get_annotated_sentences(annotated_sentences):
    sentences = {}
    for topic, doc, sentence in annotated_sentences:
        if topic not in sentences:
            sentences[topic] = {}
        doc_name = f"{topic}_{doc}.xml"
        if doc_name not in sentences[topic]:
            sentences[topic][doc_name] = []
        sentences[topic][doc_name].append(sentence)
    return sentences


def prepare_dataset(data_dir: str, output_dir: str) -> None:
    # check if data directory exists
    if not (os.path.exists(data_dir) and os.path.isdir(data_dir)):
        raise Exception("Data directory does not exist")

    # make output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_path = os.path.join(data_dir, 'ECB+_LREC2014')
    zip_path = os.path.join(data_dir, 'ECB+_LREC2014', 'ECB+.zip')

    if not os.path.exists(zip_path):
        raise Exception("ECB+ dataset not found in data directory")

    try:
        subprocess.run(['unzip', '-o', zip_path, '-d', dataset_path], shell=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise Exception("Error while unzipping ECB+ dataset")

    validated_sentences = np.genfromtxt(os.path.join(dataset_path, 'ECBplus_coreference_sentences.csv'),
                                        delimiter=',', dtype=str, skip_header=1)

    validated_sentences = get_annotated_sentences(validated_sentences)

    events, entities, sentences, vocab = get_data(dataset_path, validated_sentences)

    with open(os.path.join(output_dir, 'event_gold_mentions.json'), 'w+') as f:
        json.dump(events, f)

    with open(os.path.join(output_dir, 'entity_gold_mentions.json'), 'w+') as f:
        json.dump(entities, f)

    with open(os.path.join(output_dir, 'sentences.json'), 'w+') as f:
        json.dump(sentences, f)

    with open(os.path.join(output_dir, 'vocab.txt'), 'w') as f:
        for word in vocab:
            f.write(word + '\n')


@hydra.main(version_base=None, config_path="../../conf", config_name="train")
def main(cfg: DictConfig):
    data_dir = cfg['dataset']['path']
    output_dir = cfg['dataset']['output']
    prepare_dataset(data_dir, output_dir)


if __name__ == '__main__':
    main()
