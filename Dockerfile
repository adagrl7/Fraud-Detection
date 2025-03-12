# Use an official Python runtime as a parent image
FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
# Ensure python-multipart is installed
RUN pip install python-multipart
# Copy the rest of your application code, including the templates folder
COPY . .

EXPOSE 8001
ENV PORT=8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
