python -m venv .venv

Set-ExecutionPolicy Unrestricted -Scope CurrentUser
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r ./requirements.txt
pip install -e .