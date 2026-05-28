FROM python:3.12-slim
WORKDIR /app
RUN pip install mlflow==2.19.0 scikit-learn pandas numpy
COPY MLProject/ ./MLProject/
EXPOSE 5001
CMD ["python", "MLProject/modelling.py"]