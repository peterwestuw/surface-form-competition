mkdir data/cb
cd data/cb
wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/CB.zip
unzip CB.zip
cp CB/val.jsonl dev.jsonl
