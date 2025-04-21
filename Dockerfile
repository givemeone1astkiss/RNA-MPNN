FROM  pytorchlightning/pytorch_lightning
LABEL authors="Huazhi Wang"


#RUN mkdir /saisdata/ \
#    && mkdir /saisresult/

#ADD ./data/coords/ /saisdata/coords/
#ADD ./data/seqs/ /saisdata/seqs/
ADD ./rnampnn /app/rnampnn
ADD ./out /app/out
ADD ./run.sh ./main.py /app/
ADD ./env/ /app/env/
WORKDIR /app
RUN cd /app \
    && pip install numpy biopython pandas tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install env/torch_scatter-2.1.2+pt25cu124-cp312-cp312-linux_x86_64.whl
CMD ["sh", "/app/run.sh"]