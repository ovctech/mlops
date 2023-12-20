TARGET = mlops
SRC = mlops/*
OS = $(shell uname)
ifeq ($(OS), Linux)
	COLOR = \e
else
	COLOR = \033
endif

all: pull-data

pull-data:
	@-echo "$(COLOR)[32m Pulling data... $(COLOR)[0m"
	@dvc pull

delete-data:
	@-echo "$(COLOR)[32m Deleting data... $(COLOR)[0m"
	@rm -rf data/cifar-10-batches-py

clean:
	@rm -rf tests/test_results obj *.dSYM **/*.dSYM report_html/ test_coverage.info 3d_functions/*.o
	@-echo "$(COLOR)[32m CLEAN... SUCCESS $(COLOR)[0m"

fclean: clean
	@rm -rf ../build obj_files ../dist $(HOME)/Desktop/$(TARGET).app ../.vscode/ application/application.pro.user tests/*.gcno
	@-echo "$(COLOR)[32m FULL CLEAN... SUCCESS $(COLOR)[0m"
