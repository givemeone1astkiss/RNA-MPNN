FROM  pytorchlightning/pytorch_lightning
LABEL authors="Huazhi Wang"

ADD . /app
WORKDIR /app
RUN cd /app
RUN pip install numpy biopython pandas tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
CMD ["sh", "/app/run.sh"]