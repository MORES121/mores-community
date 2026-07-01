FROM paddlepaddle/paddle:2.6.2

RUN apt-get update && apt-get install -y dos2unix && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY run_mores_clean.py /app/run_mores.py
COPY run.sh .
COPY train_det_full.py .
COPY checkpoints/det_model_epoch20.pdparams /app/model.pdparams

RUN dos2unix /app/run.sh
RUN pip install opencv-python-headless tqdm

RUN chmod +x /app/run.sh

CMD ["/app/run.sh"]