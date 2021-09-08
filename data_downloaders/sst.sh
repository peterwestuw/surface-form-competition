mkdir data/sst-2/
mkdir data/sst-5/
cd data/sst-2/
wget https://raw.githubusercontent.com/prrao87/fine-grained-sentiment/master/data/sst/sst_dev.txt -O dev.tsv
wget https://raw.githubusercontent.com/prrao87/fine-grained-sentiment/master/data/sst/sst_test.txt -O test.tsv
wget https://raw.githubusercontent.com/prrao87/fine-grained-sentiment/master/data/sst/sst_train.txt -O train.tsv
cd ../sst-5/
cp ../sst-2/{dev,test,train}.tsv .
