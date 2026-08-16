FROM tensorflow/tensorflow:2.9.3-gpu AS mic-opcc

WORKDIR /home/mic-opcc

RUN apt update
RUN apt install libpcl-dev -y
RUN pip install \
keras==2.9.0 \
numba==0.58.1 \
scipy==1.10.1 \
python-pcl==0.3.0rc1 \
tensorflow-probability==0.15.0 \
tensorflow-compression==2.9.2

COPY . .
ENTRYPOINT ["python", "./run_mic_pcc.py"]
CMD ["-X", "./samples/mini_index.txt", "-T ./samples/mini_index.txt"]