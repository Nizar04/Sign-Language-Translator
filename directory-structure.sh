#!/bin/bash

# Create main project directories
mkdir -p sign-language-translator/{src,frontend,models,data,scripts,tests}

# Create backend structure
mkdir -p sign-language-translator/src/{translator,models,utils}
touch sign-language-translator/src/main.py
touch sign-language-translator/src/__init__.py

# Create frontend structure
mkdir -p sign-language-translator/frontend/{src,public}
mkdir -p sign-language-translator/frontend/src/{components,hooks,pages}

# Create data directories
mkdir -p sign-language-translator/data/{vocabularies,configs}

# Create configuration files
touch sign-language-translator/config.yaml
touch sign-language-translator/.env
touch sign-language-translator/.gitignore
touch sign-language-translator/requirements.txt
touch sign-language-translator/README.md
touch sign-language-translator/LICENSE

# Create test structure
mkdir -p sign-language-translator/tests/{unit,integration}
touch sign-language-translator/tests/__init__.py

# Create scripts
touch sign-language-translator/scripts/download_models.py
touch sign-language-translator/scripts/setup_database.py
