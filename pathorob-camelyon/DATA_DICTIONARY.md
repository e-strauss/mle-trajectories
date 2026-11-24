# Lymph-node image classification

Each sample is one 256 x 256 RGB microscopy patch. All exported PNGs preserve
the decoded source pixels and omit source filenames and image metadata.

| Field | Meaning |
| --- | --- |
| sample_id | Opaque unique sample identifier; never a prediction feature. |
| image_path | Path relative to input/, e.g. images/img_....png. |
| label | 0 = normal tissue; 1 = tumor tissue; development data only. |
| slide_id | Opaque slide identifier; all its patches belong to one split. |
| group_id | Opaque grouping key for leakage-aware splitting. |
| center_id | Opaque medical-center identifier; development data only. |
| tumor_probability | Model probability of label 1, in [0, 1]. |

The fixed train and validation partitions are balanced within each of the two
development centers. Additional splits must keep each group intact. Patient
linkage is unavailable for the development slides: group_id represents slide
identity, not a verified distinct patient. Do not claim patient-independent
validation from these identifiers alone.

Test IDs reveal neither the label nor the evaluation subset. The final scorer
separately evaluates held-out slides from development centers and unseen-center
slides; the latter balanced accuracy is the primary score. Validation uses the
same formula and fixed 0.5 decision threshold, through metric.py.

Identifiers are for joining and splitting, not for learning the target.
The labels and source mappings for test samples are unavailable in input/.
