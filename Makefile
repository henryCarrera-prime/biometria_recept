# Makefile para gestión de proyecto Biometria

SHELL := /bin/bash
PYTHON_VERSION := 3.10.5
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

.PHONY: all setup install run migrate clean prod test lint help check_pyenv install_pyenv

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

	export PYENV_ROOT="$$HOME/.pyenv"; \
	export PATH="$$PYENV_ROOT/bin:$$PATH"; \
	eval "$$(pyenv init --path)"; \
	eval "$$(pyenv init -)"; \

	pyenv install -s $(PYTHON_VERSION)
	pyenv local $(PYTHON_VERSION)

	python3 -m venv $(VENV_DIR)
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
	$(VENV_DIR)/bin/gunicorn core.wsgi:application --bind 0.0.0.0:8000 --workers 3 --timeout 120

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