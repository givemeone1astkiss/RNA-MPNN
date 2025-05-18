FROM  pytorchlightning/pytorch_lightning
LABEL authors="Huazhi Wang"

ADD rnampnn /app/rnampnn
ADD rdesign /app/rdesign
ADD ./out /app/out
ADD ./run.sh ./main.py /app/
WORKDIR /app
RUN cd /app \
    && pip install numpy biopython pandas tqdm xgboost scikit-learn seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple
CMD ["sh", "/app/run.sh"]