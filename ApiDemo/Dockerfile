FROM python:3.7-alpine

RUN apk add --no-cache nginx

WORKDIR /src

COPY /src/requirements.txt /src

RUN pip install --no-deps -r requirements.txt

COPY /src /src

COPY api_gunicorn_docker_launch.sh /

COPY nginx.conf /etc/nginx/conf.d/default.conf

RUN mkdir -p /run/nginx && nginx -t

EXPOSE 8080

ENTRYPOINT ["/api_gunicorn_docker_launch.sh"]
