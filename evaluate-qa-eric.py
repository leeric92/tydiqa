""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
import collections
from collections import Counter
import string
import re
import argparse
import json
import sys
import os
import statistics


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return {
            'f1': 0,
            'precision': 0,
            'recall': 0
        }

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)



def metric_max_over_ground_truths_2(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)

    def func(p):
        return p['f1']

    return max(scores_for_ground_truths, key=func)


def evaluate(dataset, predictions, labels_classifier_file=None, f1_threshold=0.5):
    f1 = exact_match = total = precision = recall = 0
    labels_classifier = collections.defaultdict(dict)
    labels = collections.defaultdict(dict)
    correct = 0
    incorrect = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]

                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)

                obj_score = metric_max_over_ground_truths_2(
                    f1_score, prediction, ground_truths)

                f1 += obj_score['f1']
                precision += obj_score['precision']
                recall += obj_score['recall']

                labels[qa['id']] = {}
                labels[qa['id']]['f1'] = obj_score['f1']
                labels[qa['id']]['ground_truths'] = ground_truths
                labels[qa['id']]['prediction'] = prediction

    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                qa_f1 = labels[qa['id']]['f1']

                labels_classifier[qa['id']] = dict()
                labels_classifier[qa['id']]['context'] = paragraph['context']
                labels_classifier[qa['id']]['qa'] = qa
                labels_classifier[qa['id']]['f1'] = qa_f1
                labels_classifier[qa['id']]['ground_truths'] = labels[qa['id']]['ground_truths']
                labels_classifier[qa['id']]['prediction'] = labels[qa['id']]['prediction']

                if labels_classifier_file != None:
                    if qa_f1 >= float(f1_threshold):
                        labels_classifier[qa['id']]['label'] = 0 ## correct
                        correct += 1
                    else:
                        labels_classifier[qa['id']]['label'] = 1 ## not correct
                        incorrect += 1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    precision = 100.0 * precision / total
    recall = 100.0 * recall / total

    if labels_classifier_file != None:
        if os.path.exists(labels_classifier_file):
            with open(labels_classifier_file) as fp:
                labels_classifier_data = json.load(fp)
                labels_classifier.update(labels_classifier_data)
                
        with open(labels_classifier_file, 'w') as fp:
            json.dump(labels_classifier, fp)
    return {'exact_match': exact_match, 'f1': f1, 'correct': correct, 'incorrect': incorrect, 'precision': precision, 'recall': recall}


if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    parser.add_argument('labels_classifier_file', help='Labels classifier file')
    parser.add_argument('f1_threshold', help='F1 threshold')
    args = parser.parse_args()
    
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    #labels_classifier_file = json.load(args.labels_classifier_file)
    print(json.dumps(evaluate(dataset, predictions, args.labels_classifier_file, args.f1_threshold)))
