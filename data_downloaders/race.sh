mkdir data/race-m/
cd data/race-m/
wget http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz
tar -xvf RACE.tar.gz
cp -r RACE/{dev,test} .
cd ..
mkdir race-h
cd race-h
cp -r ../race-m/{dev,test} .
