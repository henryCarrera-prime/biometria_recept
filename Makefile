# Makefile para gestión de proyecto Biometria

SHELL := /bin/bash
PYTHON_VERSION := 3.10.5
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
GUNICORN_WORKERS ?= 2
GUNICORN_TIMEOUT ?= 120
GUNICORN_BIND ?= 0.0.0.0:8000
APP_NAME ?= biometria
APP_USER ?= ubuntu
APP_DIR ?= /home/ubuntu/biometria
GUNICORN_INTERNAL_BIND ?= 127.0.0.1:8000
SYSTEMD_SERVICE ?= $(APP_NAME).service
NGINX_SITE ?= $(APP_NAME)

.PHONY: all setup install run migrate clean prod test lint help check_pyenv install_pyenv \
	systemd_install systemd_restart systemd_status nginx_install nginx_status deploy_http80

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================
# PYENV
# =============================

check_pyenv: ## Verifica si pyenv está instalado
	@if ! command -v pyenv >/dev/null 2>&1; then \
		echo "pyenv no encontrado..."; \
		$(MAKE) install_pyenv; \
	else \
		echo "pyenv ya está instalado"; \
	fi

install_pyenv: ## Instala pyenv
	@echo "Instalando dependencias del sistema..."
	sudo apt update
	sudo apt install -y \
	build-essential \
	make \
	gcc \
	libssl-dev \
	zlib1g-dev \
	libbz2-dev \
	libreadline-dev \
	libsqlite3-dev \
	curl \
	llvm \
	libncursesw5-dev \
	xz-utils \
	tk-dev \
	libxml2-dev \
	libxmlsec1-dev \
	libffi-dev \
	liblzma-dev \
	libgl1
	@echo "Instalando pyenv..."
	curl https://pyenv.run | bash
	@echo 'export PYENV_ROOT="$$HOME/.pyenv"' >> $$HOME/.bashrc
	@echo 'export PATH="$$PYENV_ROOT/bin:$$PATH"' >> $$HOME/.bashrc
	@echo 'eval "$$(pyenv init --path)"' >> $$HOME/.bashrc
	@echo 'eval "$$(pyenv init -)"' >> $$HOME/.bashrc
	@echo "Ejecuta: source ~/.bashrc o reinicia la sesión"

# =============================
# SETUP
# =============================

setup: check_pyenv ## Configuración inicial
	@echo "Configurando entorno con Python $(PYTHON_VERSION)..."
	@export PYENV_ROOT="$$HOME/.pyenv"; \
	export PATH="$$PYENV_ROOT/bin:$$PATH"; \
	if [ ! -x "$$PYENV_ROOT/bin/pyenv" ]; then \
		echo "pyenv no encontrado en $$PYENV_ROOT/bin/pyenv"; \
		echo "Ejecuta: source ~/.bashrc o abre una nueva sesión y vuelve a correr make setup"; \
		exit 1; \
	fi; \
	"$$PYENV_ROOT/bin/pyenv" install -s $(PYTHON_VERSION); \
	"$$PYENV_ROOT/bin/pyenv" local $(PYTHON_VERSION); \
	PYENV_VERSION=$(PYTHON_VERSION) "$$PYENV_ROOT/bin/pyenv" exec python -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@if [ ! -f .env ]; then \
		echo "Creando archivo .env básico..."; \
		echo "LOG_LEVEL=INFO" > .env; \
		echo "USE_HTTPS=False" >> .env; \
	fi

	@echo "Entorno configurado correctamente."

# =============================
# OPERACIONES
# =============================

install: ## Instala dependencias
	$(PIP) install -r requirements.txt

migrate: ## Ejecuta migraciones
	$(PYTHON) manage.py migrate

run: ## Servidor desarrollo
	$(PYTHON) manage.py runserver 0.0.0.0:8000

prod: ## Servidor producción (Gunicorn)
	@echo "Iniciando Gunicorn..."
	$(VENV_DIR)/bin/gunicorn core.wsgi:application --bind $(GUNICORN_BIND) --workers $(GUNICORN_WORKERS) --timeout $(GUNICORN_TIMEOUT)

clean: ## Limpia caché
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov

test: ## Pruebas
	$(PYTHON) manage.py test

lint: ## Análisis código
	$(VENV_DIR)/bin/flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	$(VENV_DIR)/bin/flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

superuser: ## Crear superusuario
	$(PYTHON) manage.py createsuperuser

shell: ## Shell Django
	$(PYTHON) manage.py shell

# =============================
# DESPLIEGUE (SYSTEMD + NGINX)
# =============================

systemd_install: ## Crea/actualiza servicio systemd de Gunicorn
	@echo "Instalando servicio $(SYSTEMD_SERVICE)..."
	@printf '%s\n' \
	'[Unit]' \
	'Description=Gunicorn daemon for $(APP_NAME)' \
	'After=network.target' \
	'' \
	'[Service]' \
	'User=$(APP_USER)' \
	'Group=www-data' \
	'WorkingDirectory=$(APP_DIR)' \
	'Environment="PATH=$(APP_DIR)/$(VENV_DIR)/bin"' \
	'EnvironmentFile=-$(APP_DIR)/.env' \
	'ExecStart=$(APP_DIR)/$(VENV_DIR)/bin/gunicorn core.wsgi:application --bind $(GUNICORN_INTERNAL_BIND) --workers $(GUNICORN_WORKERS) --timeout $(GUNICORN_TIMEOUT)' \
	'Restart=always' \
	'RestartSec=3' \
	'' \
	'[Install]' \
	'WantedBy=multi-user.target' | sudo tee /etc/systemd/system/$(SYSTEMD_SERVICE) > /dev/null
	@sudo systemctl daemon-reload
	@sudo systemctl enable $(SYSTEMD_SERVICE)
	@sudo systemctl restart $(SYSTEMD_SERVICE)
	@sudo systemctl --no-pager --full status $(SYSTEMD_SERVICE) | head -n 20

systemd_restart: ## Reinicia servicio systemd de Gunicorn
	@sudo systemctl restart $(SYSTEMD_SERVICE)
	@sudo systemctl --no-pager --full status $(SYSTEMD_SERVICE) | head -n 20

systemd_status: ## Estado del servicio systemd de Gunicorn
	@sudo systemctl --no-pager --full status $(SYSTEMD_SERVICE)

nginx_install: ## Instala y configura Nginx en puerto 80 (sin dominio)
	@echo "Instalando y configurando Nginx..."
	@sudo apt update
	@sudo apt install -y nginx
	@printf '%s\n' \
	'server {' \
	'    listen 80 default_server;' \
	'    listen [::]:80 default_server;' \
	'    server_name _;' \
	'    client_max_body_size 25M;' \
	'' \
	'    location / {' \
	'        proxy_pass http://$(GUNICORN_INTERNAL_BIND);' \
	'        proxy_http_version 1.1;' \
	'        proxy_set_header Host $$host;' \
	'        proxy_set_header X-Real-IP $$remote_addr;' \
	'        proxy_set_header X-Forwarded-For $$proxy_add_x_forwarded_for;' \
	'        proxy_set_header X-Forwarded-Proto $$scheme;' \
	'        proxy_read_timeout 120s;' \
	'    }' \
	'}' | sudo tee /etc/nginx/sites-available/$(NGINX_SITE) > /dev/null
	@sudo ln -sf /etc/nginx/sites-available/$(NGINX_SITE) /etc/nginx/sites-enabled/$(NGINX_SITE)
	@sudo rm -f /etc/nginx/sites-enabled/default
	@sudo nginx -t
	@sudo systemctl enable nginx
	@sudo systemctl restart nginx
	@sudo systemctl --no-pager --full status nginx | head -n 20

nginx_status: ## Estado de Nginx
	@sudo systemctl --no-pager --full status nginx

deploy_http80: systemd_install nginx_install ## Levanta Gunicorn como servicio y expone por Nginx:80
	@echo "Despliegue completado. API publicada en http://IP_DEL_SERVIDOR/"

