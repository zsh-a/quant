docker run \
    -p 8086:8086 \
    -v "$PWD/db:/var/lib/influxdb2" \
    -v "$PWD/config:/etc/influxdb2" \
    influxdb:2.7.6-alpine
