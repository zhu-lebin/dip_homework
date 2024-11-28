FILE=facades
URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/
mkdir -p $TARGET_DIR
echo "Downloading $URL dataset..." to $TARGET_DIR
# 设置代理
export https_proxy="210.45.73.38:8080"
export http_proxy="210.45.73.38:8080"
wget -N $URL -O $TAR_FILE
# 清理代理设置
unset http_proxy
unset https_proxy

mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE

find "${TARGET_DIR}train" -type f -name "*.jpg" |sort -V > ./train_list.txt
find "${TARGET_DIR}val" -type f -name "*.jpg" |sort -V > ./val_list.txt
