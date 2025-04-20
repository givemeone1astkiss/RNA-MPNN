FROM  pytorchlightning/pytorch_lightning
LABEL authors="Huazhi Wang"

ADD . /app
WORKDIR /app
RUN cd /app
RUN pip install numpy biopython pandas tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install torch_scatter-2.1.2+pt26cu124-cp312-cp312-linux_x86_64.whl
CMD ["sh", "/app/run.sh"]