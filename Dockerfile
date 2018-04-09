FROM floydhub/dl-base:3.0.0-gpu-py3.22
ENV APP_DIR /app
WORKDIR $APP_DIR
COPY . $APP_DIR
RUN pip install -r requirements.txt