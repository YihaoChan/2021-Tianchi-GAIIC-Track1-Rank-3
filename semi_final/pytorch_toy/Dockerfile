FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.6-cuda10.1-py3

RUN pip install tqdm \
                pandas \
                scikit-learn \
                transformers==4.4.2 \
                -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

ADD . /
WORKDIR /

CMD ["sh", "run.sh"]
