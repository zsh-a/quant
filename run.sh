export INFLUXDB_TOKEN=ct2FUk-NV8TaHBtfb-qaB2agTqv26YbSpsALL513sTui0_spDwXNTHMXokfTKbMafFUSFPAoFlxtuUdljnYvtA==
tmux new-session -d -s db './start_db.sh'
sleep 10
tmux new-session -s db_update 'python db.py'
tmux new-session -d -s live 'python model.py'
