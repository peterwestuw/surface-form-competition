mkdir data/arc-easy/
mkdir data/arc-challenge/
cd data/arc-easy/
wget https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip
unzip ARC-V1-Feb2018.zip
cp ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Dev.jsonl dev.jsonl
cp ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Test.jsonl test.jsonl
cd ../arc-challenge/
cp ../arc-easy/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Dev.jsonl dev.jsonl
cp ../arc-easy/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Test.jsonl test.jsonl
