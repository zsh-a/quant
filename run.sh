# export INFLUXDB_TOKEN=a8XLcYAfQZsftssSdOdhNeRrwAyw5k0Dv2-J2LmiXtq2LCrvfOFa84C4HiH6OIlMTSoXkDj_wyxXwqWk6qNiUw==
export INFLUXDB_TOKEN=vH5FD5il70h2n5RNO9zj6i6dRO9TMQihKWL9xDhdbarA7wyXZNM-GOgkc6MKJS3zsmYEOBaW_gylF-XVZBSR0A==
tmux new-session -d -s db './start_db.sh'
sleep 10
tmux new-session -s db_update 'python db.py'
tmux new-session -d -s live 'python model.py'
