FROM  registry.cn-shanghai.aliyuncs.com/tcc_public/python:3.10
LABEL authors="Huazhi Wang"

ADD . /app
WORKDIR /app
RUN cd /app
RUN pip install -r ./requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
CMD ["python", "main.py"]
ENTRYPOINT ["top", "-b"]