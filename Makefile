TARGET = mlops
PYTHON_V = 3.11.0
OS = $(shell uname)
ifeq ($(OS), Linux)
	COLOR = \e
else
	COLOR = \033
endif

all: train infern

rebuild: clean all

prepare: pre-install create-venv install open-mlflow

pre-install:
	@-echo "$(COLOR)[32m Installing pyenv... $(COLOR)[0m"
	@sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl
	@curl https://pyenv.run | bash
	@export PATH="$HOME/.pyenv/bin:$PATH"
	@eval "$(pyenv init -)"
	@eval "$(pyenv virtualenv-init -)"
	@exec "$SHELL"

create-venv:
	@-echo "$(COLOR)[32m Creating clean venv... $(COLOR)[0m"
	@pyenv install $(PYTHON_V)
	@pyenv virtualenv $(PYTHON_V) $(TARGET)
	@pyenv local $(PYTHON_V)/envs/$(TARGET)
	@pip install poetry
	@poetry shell

install:
	@-echo "$(COLOR)[32m Installing libs... $(COLOR)[0m"
	@poetry install
	@pre-commit install
	@-echo "$(COLOR)[32m Pre-commit checks... $(COLOR)[0m"
	@pre-commit run -a

train:
	@-echo "$(COLOR)[32m Training... $(COLOR)[0m"
	@cd $(TARGET) && python3 trainer.py && cd ..

open-mlflow:
	gnome-terminal -- bash -c "mlflow server --host 127.0.1.1 --port 8080"
	xdg-open http://127.0.1.1:8080/

infern:
	@-echo "$(COLOR)[32m Infern... $(COLOR)[0m"
	@cd $(TARGET) && python3 infer.py && cd ..
	@-echo "$(COLOR)[32m Results saved at $(TARGET)/results$(COLOR)[0m"

delete-data:
	@-echo "$(COLOR)[32m Deleting data... $(COLOR)[0m"
	@rm -rf data/cifar-10-batches-py

clean: delete-data
	@-echo "$(COLOR)[32m Cleaning... $(COLOR)[0m"
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	@rm -rf $(TARGET)/lightning_logs/*
	@rm -rf $(TARGET)/results/*.csv
	@rm -rf $(TARGET)/models/*.ckpt
	@rm -rf $(TARGET)/outputs/*
	@rm -rf mlruns/
