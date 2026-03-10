FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md requirements.txt ./
COPY graphrag_studio ./graphrag_studio
COPY streamlit_app.py ./
COPY data ./data
COPY docs ./docs
COPY assets ./assets
COPY .streamlit ./.streamlit
COPY scripts ./scripts

RUN pip install --upgrade pip && pip install .

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
