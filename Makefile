build:
	docker-compose build

download-data:
	kaggle competitions download -c tgs-salt-identification-challenge

docker-bash:
	docker-compose run --rm learning-tool /bin/bash

format:
	yapf --in-place -r --verbose --exclude venv .

submit: ./learning-tool/src/predictions.csv
	cp ./learning-tool/src/predictions.csv /tmp/predictions.csv
	gzip /tmp/predictions.csv
	kaggle competitions submit -f /tmp/predictions.csv.gz -m "$(date)" -c tgs-salt-identification-challenge
	rm /tmp/predictions.csv.gz

tensorboard:
	docker-compose stop tensorboard
	docker-compose rm -f tensorboard
	docker-compose up -d tensorboard

notebook:
	docker-compose stop notebook
	docker-compose rm -f notebook
	docker-compose up -d notebook
