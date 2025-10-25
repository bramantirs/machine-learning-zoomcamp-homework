FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

COPY pipeline_v1.bin /app/
COPY app.py /app/

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" scikit-learn

EXPOSE 9696

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9696"]
