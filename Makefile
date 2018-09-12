build:
	docker-compose build

docker-bash:
	docker-compose run --rm learning-tool /bin/bash

run:
	docker-compose run --rm learning-tool python main.py

format:
	yapf --in-place -r --verbose --exclude venv .

submit: ./learning-tool/src/predictions.csv
	cp ./learning-tool/src/predictions.csv /tmp/predictions.csv
	gzip /tmp/predictions.csv
	kaggle competitions submit -f /tmp/predictions.csv.gz -m "$(date)" -c tgs-salt-identification-challenge
	rm /tmp/predictions.csv.gz

tensorbard:
	docker-compose stop tensorboard
	docker-compose rm -f tensorboard
	docker-compose up -d tensorboard
