nohup docker run \
    --volume $PWD/db:/bitnami/clickhouse \
    --env ALLOW_EMPTY_PASSWORD=yes \
    -p 9000:9000 \
    -p 8123:8123 \
    bitnami/clickhouse:latest > /dev/null 2>&1 &
