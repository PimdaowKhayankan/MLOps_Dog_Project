FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install python-multipart

COPY app.py .

EXPOSE 8087

CMD ["python", "app.py"]
