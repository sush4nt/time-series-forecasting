# How to Set Up This Project (uv + pyproject.toml + Jupyter)

## 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

## 2. Initialize the project

```bash
uv init --no-readme --python 3.11
```

Skip this step if `pyproject.toml` already exists.

## 3. Add project dependencies

```bash
uv add "numpy~=1.26.0" \
       "pandas~=2.2.0" \
       "pyarrow~=15.0.0" \
       "scikit-learn~=1.4.0" \
       "xgboost~=2.0.3" \
       "pyyaml~=6.0" \
       "mlflow~=2.11.0" \
       "onnxmltools~=1.12.0" \
       "onnxconverter-common~=1.14.0" \
       "onnxruntime~=1.17.0" \
       "fastapi~=0.110.0" \
       "uvicorn[standard]~=0.29.0" \
       "pydantic~=2.6.0" \
       "prometheus-client~=0.20.0" \
       "httpx~=0.27.0" \
       "setuptools>=69,<81"
```

This resolves and installs everything into a `.venv` in the project root and writes `uv.lock`.

## 3b. Add PyTorch (Part C — GRU, CPU build)

PyTorch is pulled from the CPU wheel index (pinned in `pyproject.toml` under
`[tool.uv.sources]` / `[[tool.uv.index]]`), so a plain `uv sync` installs it:

```bash
uv sync
```

To add/refresh it explicitly into the active venv without touching resolution:

```bash
uv pip install "torch~=2.2.0" --index-url https://download.pytorch.org/whl/cpu
```

## 4. Run commands through the environment

```bash
uv run python your_script.py
```

Or activate it directly:

```bash
source .venv/bin/activate
```

## 5. Sync the environment (e.g. on a new machine or after pulling changes)

```bash
uv sync
```

## 6. (Optional) Export a `requirements.txt` for tooling that still needs it

```bash
uv export --format requirements-txt > requirements.txt
```

## 7. Add dev-only dependencies

```bash
uv add --dev pytest ruff jupyter
```

## 8. Make the environment available in Jupyter

Add `ipykernel`:

```bash
uv add --dev ipykernel
```

Register the venv as a Jupyter kernel:

```bash
uv run python -m ipykernel install --user --name assignment-ml-engineer-v3 --display-name "Python (Assignment ML Engineer)"
```

Then select the kernel:

- **In Cursor**: use the kernel picker in the notebook UI and choose the `.venv` interpreter or the registered kernel (`Python (Assignment ML Engineer)`).
- **In classic Jupyter**: `uv run jupyter notebook`, then Kernel → Change Kernel.

Verify the active interpreter from within a notebook cell:

```python
import sys
print(sys.executable)
```

It should point to `.venv/bin/python` inside the project directory.

### Notes

- After `uv add <package>`, you generally do **not** need to re-register the kernel — the kernelspec points at the same `.venv` interpreter, so newly installed packages are picked up automatically. Restart the kernel if you already imported the package (or a failed import) earlier in the session.
- You only need to re-register the kernel if you delete/recreate the `.venv`, move/rename the project, or want to change the kernel's display name.
</contents>
</invoke>
