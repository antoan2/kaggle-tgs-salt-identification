build:
	docker-compose build

docker-bash:
	docker-compose run --rm learning-tool /bin/bash

run:
	docker-compose run --rm learning-tool python main.py

format:
	yapf --in-place -r --verbose --exclude venv .

submit: ./learning-tool/src/predictions.txt
	cp ./learning-tool/src/predictions.txt /tmp/predictions.txt
	gzip /tmp/predictions.txt
	kaggle competitions submit -f predictions.csv.gz -m "$(date)" -c tgs-salt-identification-challenge
	rm /tmp/predictions.txt.gz
