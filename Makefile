clean_models:
	@rm *.pth

clean_runs:
	@rm -rf runs/*

clean_all: clean_models clean_runs

clean_logs:
	@rm *.log

dev_local:
	./dev_local.sh up

dev_local_down:
	./dev_local.sh down

dev_local_status:
	./dev_local.sh status
