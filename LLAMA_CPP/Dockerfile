ARG FUNCTION_DIR="/function"

FROM --platform=linux/amd64 python:3.11 as build-image

ARG FUNCTION_DIR
RUN mkdir -p ${FUNCTION_DIR}
WORKDIR ${FUNCTION_DIR}

COPY model/ model/
COPY requirements.txt .
RUN apt-get update
RUN apt-get -y install libopenblas-dev
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install --target ${FUNCTION_DIR} --no-cache-dir llama-cpp-python
RUN pip install --target ${FUNCTION_DIR} --no-cache-dir -r requirements.txt

COPY lambda_function.py .

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "lambda_function.handler" ]

FROM --platform=linux/amd64 python:3.11-slim as custom-docker-runtime

ARG FUNCTION_DIR
WORKDIR ${FUNCTION_DIR}

RUN apt-get update
RUN apt-get -y install libopenblas-dev

COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "lambda_function.handler" ]

