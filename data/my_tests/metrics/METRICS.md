# Description of metrics files
 - base-lg-metrics.csv - metrics of base XTTSv2 model on [keithito/lj_speech](https://huggingface.co/datasets/keithito/lj_speech)
 - base-vp-metrics.csv - metrics of base XTTSv2 model on subset of [facebook/voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli)
 - best-lg-lg-metrics.csv - model fine tuned on [keithito/lj_speech](https://huggingface.co/datasets/keithito/lj_speech). Best model checkpoint. metrics on [keithito/lj_speech](https://huggingface.co/datasets/keithito/lj_speech)
 - last-lg-lg-metrics.csv - model fine tuned on [keithito/lj_speech](https://huggingface.co/datasets/keithito/lj_speech). Last model checkpoint. metrics on [keithito/lj_speech](https://huggingface.co/datasets/keithito/lj_speech)
 - last-lg-vp-metrics.csv - model fine tuned on [keithito/lj_speech](https://huggingface.co/datasets/keithito/lj_speech). Best model checkpoint. metrics on [facebook/voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli)

# Metrics of models:

|                                |       cer |       mer |       wer |       wil |      wip |   ref_secs |     secs |   utmos |
|:-------------------------------|----------:|----------:|----------:|----------:|---------:|-----------:|---------:|--------:|
| (4525, 'base_xtts_v2')         | 0.811699  | 0.191086  | 0.811699  | 0.219782  | 0.780218 |   0.731903 | 0.332511 | 2.46831 |
| (4525, 'last-lg-fn')           | 0.135982  | 0.0751364 | 0.135982  | 0.0940077 | 0.905992 |   0.731903 | 0.343106 | 3.15149 |
| (197469, 'base_xtts_v2')       | 0.654058  | 0.118075  | 0.654058  | 0.133622  | 0.866378 |   0.867635 | 0.298642 | 3.37068 |
| (197469, 'last-lg-fn')         | 0.0581954 | 0.0485912 | 0.0581954 | 0.0685166 | 0.931483 |   0.867635 | 0.241995 | 3.39475 |
| ('lg_speaker', 'base_xtts_v2') | 0.200779  | 0.0846433 | 0.200779  | 0.10425   | 0.89575  |   0.808919 | 0.313187 | 3.6942  |
| ('lg_speaker', 'best-lg-fn')   | 0.559562  | 0.144783  | 0.559562  | 0.156458  | 0.843542 |   0.808919 | 0.372115 | 3.69956 |
| ('lg_speaker', 'last-lg-fn')   | 0.0775391 | 0.0547185 | 0.0775391 | 0.0710923 | 0.928908 |   0.808919 | 0.335681 | 3.67354 |

## Designations
 - `4525`, `197469` - speakers from [facebook/voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli)
 - `lg_speaker` - the only speaker from [keithito/lj_speech](https://huggingface.co/datasets/keithito/lj_speech) dataset
 - `base_xtts_v2` - means was used XTTSv2 model to compute metrics
 - `best-lg-fn` - means best (base on loss) model checkpoint that was fine-tuned on [keithito/lj_speech](https://huggingface.co/datasets/keithito/lj_speech) dataset
 - `last-lg-fn` means last model checkpoint that was fine-tuned on [keithito/lj_speech](https://huggingface.co/datasets/keithito/lj_speech) dataset
