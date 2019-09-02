FROM python:3.7-slim
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN mkdir /app
WORKDIR /app

RUN mkdir model
COPY model/model.onnx model/model.onnx
COPY EmotionModel.py EmotionModel.py
EXPOSE 5000

# Define environment variable
ENV MODEL_NAME EmotionModel
ENV API_TYPE REST
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0

CMD exec seldon-core-microservice $MODEL_NAME $API_TYPE --service-type $SERVICE_TYPE --persistence $PERSISTENCE