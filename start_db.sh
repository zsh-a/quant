nohup docker run \
    --volume $PWD/db:/bitnami/clickhouse \
    --volume $(pwd)/config/backup_disk.xml:/etc/clickhouse-server/config.d/backup_disk.xml \
    --volume $PWD/backups:/backups \
    --env ALLOW_EMPTY_PASSWORD=yes \
    -p 9000:9000 \
    -p 8123:8123 \
    bitnami/clickhouse:latest > /dev/null 2>&1 &
