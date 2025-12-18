# Python image
FROM python:3.11-slim

# Working directory
WORKDIR /app

# Copy files
COPY ./model ./model
COPY ./api ./api
COPY ./app.py ./app.py
COPY ./requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000 8501

# Start FastAPI & Streamlit
CMD [ "sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0" ]
