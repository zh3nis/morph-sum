## Morpheme-aware Neural Language Model
Code for the MorphSum model from the paper "Reusing Weights in Subword-aware Neural Language Models" (submitted to NAACL 2018)

### Requirements
Code is written in Python 3 and requires TensorFlow 1.4+. It also requires the following Python modules: `numpy`, `morfessor`, `argparse`. You can install them via:
```
pip3 install numpy morfessor argparse
```

### Data
Data should be put into the `data/` directory, split into `train.txt`, `valid.txt`, and `test.txt`. Each line of the .txt file should be a sentence. The English Penn Treebank (PTB) data is given as the default.

The non-English data (Czech, French, German, Russian, and Spanish) can be downloaded from [Jan Botha's website](https://bothameister.github.io). For ease of use you can use the [Yoon Kim's script](https://github.com/yoonkim/lstm-char-cnn/blob/master/get_data.sh), which downloads these data and saves them into the relevant folders.

#### Note on non-English data
The PTB data above does not have end-of-sentence tokens for each sentence, and by default these are
appended. The non-English data already have end-of-sentence tokens for each line so, you want to add
`--eos " "` to the command line. 

### Training morfessor
To train Morfessor 2.0 on PTB run
```
mkdir morph
morfessor-train -s morph/ptb.bin data/ptb/train.txt
```

### Model
To reproduce the result of small MorphSum+RE+RW on English PTB from Table 2
```
python3 morph-sum.py
```

### Sampled Softmax
Training on a larger vocabulary will require sampled softmax (SSM) to train at a reasonable speed. You can use the `--ssm 1` option to do this.

### Other options
To see the full list of options run
```
python3 morph-sum.py -h
```