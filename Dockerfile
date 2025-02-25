FROM python:3.12

RUN pip install --upgrade pip

RUN pip install numpy
RUN pip install pillow
RUN pip install torch
RUN pip install torchaudio
RUN pip install torchvision
RUN pip install openmdao
RUN pip install matplotlib
RUN pip install packaging
RUN pip install einops
RUN pip install pandas
RUN pip install scipy
RUN pip install scikit-learn
RUN pip install cuda-python
RUN pip install torch_geometric
RUN pip install x-transformers
RUN pip install tqdm
RUN pip install pydantic
RUN pip install h5py

WORKDIR /app
COPY . /app

CMD [ "bash" ]