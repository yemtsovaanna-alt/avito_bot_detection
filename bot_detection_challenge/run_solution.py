"""Выполнить solution.ipynb с чистого Jupyter-ядра текущего Python.

Из корня репозитория:
    .venv/bin/python bot_detection_challenge/run_solution.py

Можно передать путь к копии ноутбука, рядом с которой лежит папка data.
"""

import argparse
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from jupyter_client import AsyncKernelManager
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'notebook', nargs='?', type=Path,
        default=Path(__file__).with_name('solution.ipynb'),
    )
    notebook_path = parser.parse_args().notebook.resolve()
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)

    # Старые выводы не участвуют в проверке: каждая ячейка выполняется заново.
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None

    with TemporaryDirectory(prefix='avito_notebook_') as runtime:
        runtime_path = Path(runtime)
        os.environ['IPYTHONDIR'] = str(runtime_path / 'ipython')
        kernel_environment = os.environ | {
            'JUPYTER_RUNTIME_DIR': str(runtime_path),
            'MPLCONFIGDIR': str(runtime_path / 'matplotlib'),
        }
        manager = AsyncKernelManager(
            kernel_name='python3', connection_file=str(runtime_path / 'kernel.json'),
        )
        # Используем именно интерпретатор команды, а не случайный Python из PATH.
        # Меняется объект в памяти, установленный kernelspec на диске не затрагивается.
        manager.kernel_spec.argv = [
            sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}',
        ]
        client = NotebookClient(notebook, km=manager, timeout=600, allow_errors=False)
        client.execute(
            cwd=str(notebook_path.parent), env=kernel_environment, cleanup_kc=True,
        )

    # Записываем только успешно выполненный ноутбук. Ошибка любой ячейки прервет запуск.
    nbformat.validate(notebook)
    nbformat.write(notebook, notebook_path)
    count = sum(cell.cell_type == 'code' for cell in notebook.cells)
    print(f'Выполнено ячеек с кодом: {count}')
    print(f'Ноутбук: {notebook_path}')
    print(f'Результат: {notebook_path.parent / "submission.csv"}')


if __name__ == '__main__':
    main()
