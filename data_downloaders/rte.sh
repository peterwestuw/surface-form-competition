mkdir data/rte
cd data/rte
wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/RTE.zip
unzip RTE.zip
cp RTE/val.jsonl dev.jsonl
