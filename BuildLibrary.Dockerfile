FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
COPY . /ImageFilterRL/source
WORKDIR /ImageFilterRL/source
RUN pip install -r requirements.txt
RUN pip wheel -e . 