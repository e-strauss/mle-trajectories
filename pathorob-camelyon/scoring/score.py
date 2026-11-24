#!/usr/bin/env python3
"""Offline evaluator. Keep this script and private/ out of the agent environment."""
import argparse
import json
from pathlib import Path

import pandas as pd

from common import write_json
from metric import binary_metrics


def score_frames(submission, labels):
    if list(submission.columns) != ['sample_id', 'tumor_probability']:
        raise ValueError('Submission must have exactly: sample_id,tumor_probability')
    for table in [submission, labels]:
        if table.sample_id.isna().any() or table.sample_id.eq('').any() or table.sample_id.duplicated().any():
            raise ValueError('Missing or duplicate sample IDs')
    if set(submission.sample_id) != set(labels.sample_id):
        missing = len(set(labels.sample_id) - set(submission.sample_id))
        extra = len(set(submission.sample_id) - set(labels.sample_id))
        raise ValueError(f'Submission IDs do not match test set: missing={missing}, extra={extra}')
    labels = labels.copy()
    labels['label'] = pd.to_numeric(labels.label, errors='raise')
    if set(labels.split) != {'test_id', 'test_ood'}:
        raise ValueError('Private labels must contain both test_id and test_ood')
    aligned = labels.merge(submission, on='sample_id', how='left', validate='one_to_one')
    aligned['tumor_probability'] = pd.to_numeric(aligned.tumor_probability, errors='raise')
    # Validate every probability before computing subgroup metrics.
    binary_metrics(aligned.label.to_numpy(), aligned.tumor_probability.to_numpy())
    by_split = {name: binary_metrics(group.label.to_numpy(), group.tumor_probability.to_numpy())
                for name, group in aligned.groupby('split', sort=True)}
    by_center = {name: binary_metrics(group.label.to_numpy(), group.tumor_probability.to_numpy())
                 for name, group in aligned.groupby('medical_center', sort=True)}
    return {'primary_metric': 'test_ood.balanced_accuracy',
            'primary_score': by_split['test_ood']['balanced_accuracy'],
            'threshold': 0.5,
            'by_split': by_split,
            'by_center': by_center,
            'id_minus_ood_balanced_accuracy': by_split['test_id']['balanced_accuracy'] - by_split['test_ood']['balanced_accuracy']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('submission', type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        submission = pd.read_csv(args.submission, dtype={'sample_id': str}, keep_default_na=False)
        labels = pd.read_csv(args.root / 'private/test_labels.csv', dtype={'sample_id': str}, keep_default_na=False)
        result = score_frames(submission, labels)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'Invalid submission or scoring inputs: {exc}\n')
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
