clean_models:
	@rm *.pth

clean_runs:
	@rm -rf runs/*

clean_all: clean_models clean_runs

clean_logs:
	@rm *.log
