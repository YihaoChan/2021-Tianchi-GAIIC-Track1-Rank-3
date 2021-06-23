FROM registry.cn-shanghai.aliyuncs.com/tcc-public/tensorflow:1.13.1-cuda10.0-py3

RUN pip install tqdm \
                h5py==2.10.0 \
                pandas \
                tensorflow-gpu==1.15.0 \
                torch==1.4.0 \
                bert4keras==0.10.0 \
                scikit-learn \
                iterative-stratification==0.1.6 \
                -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

ADD . /
WORKDIR /

CMD ["sh", "run.sh"]

