FROM gcr.io/tpu-pytorch/xla:r1.9_3.8_tpuvm

RUN python3 -m pip install --no-cache-dir --upgrade google-cloud-storage torchmetrics

ENV XRT_TPU_CONFIG="localservice;0;localhost:51011"
ENV XLA_USE_BF16=1

WORKDIR /root/app/

COPY . .

ENTRYPOINT ["python3", "tpu_main.py"]