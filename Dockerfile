FROM tensorflow/tensorflow:2.9.0-gpu as mic-opcc

WORKDIR /mic-opcc

RUN pip install \
keras==2.9.0 \
numba==0.60 \
scipy==1.12.0 \
tensorflow-probability==0.15.0 \
tensorflow-compression==2.9.2

COPY . .
ENTRYPOINT ["python", "./train_mic_pcc.py"]
CMD []