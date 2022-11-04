# Dockerfile

# FROM directive instructing base image to build upon
FROM python:3.8-slim-buster

# install nginx
RUN apt-get update && apt-get install nginx vim -y --no-install-recommends
COPY nginx.default /etc/nginx/sites-available/default
RUN ln -sf /dev/stdout /var/log/nginx/access.log \
    && ln -sf /dev/stderr /var/log/nginx/error.log


# copy source and install dependencies
RUN mkdir -p /opt/app
RUN mkdir -p /opt/app/pip_cache

# Install requirements
COPY requirements.txt start-server.sh /opt/app/
COPY . /opt/app/
WORKDIR /opt/app
RUN pip install -r requirements.txt --cache-dir /opt/app/pip_cache
RUN chown -R www-data:www-data /opt/app
RUN chmod u+x start-server.sh
# start server
EXPOSE 50004
STOPSIGNAL SIGTERM
CMD ["/opt/app/start-server.sh"]

