# Source Code for Content Word Filtering and Speaker Preference Model

## Dependencies
 * Java 8 is required.

## Input Data
An input data file should be a CSV file, where each line represents an utterance. This file must have the following columns:
 * `SeqId`: the ID of the conversation to which the utterance belongs. `SeqId` can be any string and must be unique for each conversation.
 * `InstNo`: the index of the utterance. `InstNo` can be any integer and must be unique within a conversation. In case multi-level structure is not used, utterances within each conversation are ordered by `InstNo`.
 * `Author`: the speaker of the utterance. `Author` can be any string.
 * `Text`: the text of the utterance. If text is pre-tokenized, the tokens must be separated by spaces, and sentences by `<SENT>`. For example, the sentence `I'm a student. You're a teacher.` may be represented as `I 'm a student . <SENT> You 're a teacher .`. If text is not pre-tokenized and the model is given the `-tok` option, the model will automatically conduct sentence segmentation and tokenization.
 * `Parent` (optional): the `InstNo` of the parent of the utterance. This column is ignored if the model is given the `-seq` option.
 * `Domain` (optional): the domain of the utterance. This column must be present if the model is given the `-domain` option.
 * `Label` (optional): the true label (dialogue act) of the utterance. If `Label` exists, the output `InstAssign` file has the `Label` column filled.


## Command Examples

### Training
A new model is trained based on the given input data.

```
-s 5 -ft 10 -bt 10 -fa 0.1 -ba 1 -b 0.001 -ag 0.1 -sg 1 -e 0.8 -n 0.9 -mw 1 -ms 1 -seq 
-d data_dir -data data_train.csv -i 1000 -to 100 -log 100 -th 2
```

### Fitting
A trained model is fitted to the given input data, without parameter updates. This mode may be used to estimate the states of new data. A configuration passed along as input arguments is ignored, as the code automatically parses the configuration from the model path.

```
-d data_dir -data data_test.csv -i 1000 -to 100 -log 100 -th 2 
-model data_dir/CSM-data_train-S5-FT10-BT10-FA0.1-BA1.0-B0.001-SG1.0-AG0.1-E0.8-N0.9-SEQ/I1000
```

You can also generate multiple samples of state assignment every `sample` iterations after `burn` iterations (rather than just after the last iteration).

```
-d data_dir -data data_test.csv -i 1000 -to 100 -log 100 -th 2 
-model data_dir/CSM-data_train-S5-FT10-BT10-FA0.1-BA1.0-B0.001-SG1.0-AG0.1-E0.8-N0.9-SEQ/I1000 
-burn 800 -sample 20
```


## Tips
 * We recommend NOT using stop words, because exhaustive lists of stop words may remove important words from the vocabulary, such as "should", "what", etc.


