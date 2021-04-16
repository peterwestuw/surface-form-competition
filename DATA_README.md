# Instructions for Downloading Data

Here we include instructions for downloading each dataset, for maximum reproducibility. Note that we do not download/process test sets where numbers are reported on the dev set, in order to follow previous work; please see the paper for details.

## Directory Structure

The location of datasets is hard-coded in `score.py`; this is a feature, not a bug, because it is easy to lose track of which data files have what in them. All datasets are stored in `data/<dataset_name>/`. To apply to a new dataset, add a data loader to `data_loaders.py` and add a couple lines of logic code to `score.py`. There are more than 20 examples of data loaders and logic code in the respective files.

## Datasets

### COPA

The official website for COPA is: https://people.ict.usc.edu/~gordon/copa.html and you can download copa using:

```
mkdir data/copa/
cd data/copa/
wget https://people.ict.usc.edu/~gordon/downloads/COPA-resources.tgz
tar -xvf COPA-resources.tgz
cp COPA-resources/datasets/copa-{dev,test}.xml .
```

or simply use `bash data_downloaders/copa.sh`

### StoryCloze/RocStories

The official website of StoryCloze is https://cs.rochester.edu/nlp/rocstories/ You can download the validation and test files using:

```
mkdir data/storycloze/
cd data/storycloze/
wget https://github.com/snigdhac/StoryComprehension_EMNLP/raw/master/Dataset/RoCStories/cloze_test_val__spring2016%20-%20cloze_test_ALL_val.tsv -O dev.tsv
wget https://github.com/snigdhac/StoryComprehension_EMNLP/raw/master/Dataset/RoCStories/test_spring2016.tsv -O test.tsv
```

or simply use `bash data_downloaders/storycloze.sh`

### HellaSwag

The official website of HellaSwag is https://rowanzellers.com/hellaswag/ You can download the validation and test files using:

```
mkdir data/hellaswag/
cd data/hellaswag/
wget https://github.com/rowanz/hellaswag/raw/master/data/hellaswag_val.jsonl -O dev.jsonl
wget https://github.com/rowanz/hellaswag/raw/master/data/hellaswag_test.jsonl -O test.jsonl
```

or simply use `bash data_downloaders/hellaswag.sh`

### RACE-M & RACE-H

The official website of RACE: https://www.cs.cmu.edu/~glai1/data/race/ Please sign the dataset release form: https://forms.gle/XFDvLGbpS4EqLQh28

You can then download RACE using:

```
mkdir data/race-m/
cd data/race-m/
wget http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz -O dev.jsonl
tar -xvf RACE.tar.gz
cp RACE/{dev,test} .
cd ..
mkdir race-h
cd race-h
cp ../race-m/{dev,test} .
```

or simply use `bash data_downloaders/race.sh`

### ARC-Easy and ARC-Challenge

The official website of ARC is: https://allenai.org/data/arc You can download ARC-Easy and ARC-Challenge using:

```
mkdir data/arc-easy/
mkdir data/arc-challenge/
cd data/arc-easy/
wget https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip -O dev.jsonl
cp ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Dev.jsonl dev.jsonl
cp ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Test.jsonl test.jsonl
cd ../arc-challenge/
cp ../arc-easy/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Dev.jsonl dev.jsonl
cp ../arc-easy/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Test.jsonl test.jsonl
```

or simply use `bash data_downloaders/arc.sh`

### OpenBookQA

The official website of OpenBookQA is: https://allenai.org/data/open-book-qa You can download OpenBookQA using:

```
mkdir data/obqa
cd data/obqa
wget https://ai2-public-datasets.s3.amazonaws.com/open-book-qa/OpenBookQA-V1-Sep2018.zip
unzip OpenBookQA-V1-Sep2018.zip
cp OpenBookQA-V1-Sep2018/Data/Main/dev.jsonl dev.jsonl
cp OpenBookQA-V1-Sep2018/Data/Main/test.jsonl test.jsonl
```

or simply use `bash data_downloaders/obqa.sh`

### CommonsenseQA

The official website of CommonsenseQA is https://www.tau-nlp.org/commonsenseqa You can download the validation file using:

```
mkdir data/cqa/
cd data/cqa/
wget https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl -O dev.jsonl
```

or simply use `bash data_downloaders/cqa.sh`

### BoolQ

The official webpage for BoolQ is: https://github.com/google-research-datasets/boolean-questions You need to download BoolQ from Kaggle at https://www.kaggle.com/averkij/boolq-dataset/download then place the resulting `archive.zip` in `data/boolq/` and run:

```
cd data/boolq
unzip archive.zip
```

or simply use `bash data_downloaders/boolq.sh`


### RTE

We use SuperGlue's version of RTE: https://super.gluebenchmark.com/ You can download RTE using:

```
mkdir data/rte
cd data/rte
wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/RTE.zip
unzip RTE.zip
cp RTE/val.jsonl dev.jsonl
```

or simply use `bash data_downloaders/rte.sh`

### CB

We use SuperGlue's version of CB: https://super.gluebenchmark.com/ You can download CB using:

```
mkdir data/cb
cd data/cb
wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/CB.zip
unzip CB.zip
cp CB/val.jsonl dev.jsonl
```

or simply use `bash data_downloaders/cb.sh`

### Stanford Sentiment Treebank 2 & 5

The Stanford Sentiment Treebank's (SST) official website is: https://nlp.stanford.edu/sentiment/treebank.html You can download SST using:

```
mkdir data/sst-2/
mkdir data/sst-5/
cd data/sst-2/
wget https://data.deepai.org/sst5.zip
unzip sst5.zip
unzip stanfordSentimentTreebank.zip
mv stanfordSentimentTreebank/ sst
cd ../sst-5/
cp ../sst-2/sst/ .
```

or simply use `bash data_downloaders/sst.sh`

### AG's News

The official website of the AG's News corpus: http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html You can download the classification version of AG's News using:

```
mkdir data/agsnews/
cd data/agsnews/
wget https://github.com/tonyzhaozh/few-shot-learning/raw/main/data/agnews/train.csv -O dev.csv
wget https://github.com/tonyzhaozh/few-shot-learning/raw/main/data/agnews/test.csv -O test.csv
```

or simply use `bash data_downloaders/agsnews.sh`

### TREC

The official website of the TREC quesiton classification corpus: https://cogcomp.seas.upenn.edu/Data/QA/QC/ You can download the dataset using:

```
mkdir data/trec/
cd data/trec/
wget https://raw.githubusercontent.com/tonyzhaozh/few-shot-learning/main/data/trec/train.txt -O dev.txt
wget https://raw.githubusercontent.com/tonyzhaozh/few-shot-learning/main/data/trec/test.txt -O test.txt
```

or simply use `bash data_downloaders/trec.sh`
