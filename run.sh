export INFLUXDB_TOKEN=LlgUp8hZnRgoOCteZaLlJHxue0FL19VWYQbnxETmg8QgodUVoR_i97zogTcb5vViJBHOgANF5c8g0eodPxfSIw==
# export INFLUXDB_TOKEN=vH5FD5il70h2n5RNO9zj6i6dRO9TMQihKWL9xDhdbarA7wyXZNM-GOgkc6MKJS3zsmYEOBaW_gylF-XVZBSR0A==
docker rm stock_db
tmux new-session -d -s db './start_db.sh'
sleep 10
tmux new-session -s db_update 'python db.py --update'
tmux new-session -d -s live 'python run.py --live'
