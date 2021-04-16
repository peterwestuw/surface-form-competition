mkdir data/obqa
cd data/obqa
wget https://ai2-public-datasets.s3.amazonaws.com/open-book-qa/OpenBookQA-V1-Sep2018.zip
unzip OpenBookQA-V1-Sep2018.zip
cp OpenBookQA-V1-Sep2018/Data/Main/dev.jsonl dev.jsonl
cp OpenBookQA-V1-Sep2018/Data/Main/test.jsonl test.jsonl
