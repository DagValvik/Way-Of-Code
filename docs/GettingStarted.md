## Introduction to Python

Python is a versatile and widely-used programming language known for its simplicity and readability. Whether you're a beginner taking your first steps in coding or an experienced developer, Python offers tools and libraries to make development efficient and enjoyable. This guide will help you get started with Python, from installation to running your first program.

### Installing Python

#### For macOS:
1. Visit the official Python website: [python.org](https://www.python.org/downloads/macos/).
2. Download the latest Python installer for macOS.
3. Run the installer and follow the on-screen instructions to complete the setup.

#### For Windows:
1. Go to [python.org](https://www.python.org/downloads/windows/) and download the latest Python installer.
2. Run the installer, ensuring you check the "Add Python to PATH" option during installation.
3. Follow the instructions to finish the setup.

### Using the UV Package Manager

An alternative to manually managing Python installations is to use the UV package manager. This tool automates version management and simplifies project setup. Here's how to get started:

1. Install UV by following the instructions at [astral.sh/uv/](https://docs.astral.sh/uv/getting-started/installation/).
2. Use UV to install Python:
   ```sh
   uv python install
   ```
   This approach helps you manage multiple Python versions and switch between them effortlessly.
3. When starting a new project, initialize it with:
   ```sh
   uv init
   ```
   This command creates the necessary configuration files for your project.

### Managing Python Packages

Python uses `pip`, its built-in package manager, to install and manage libraries. If you are using UV, you can also manage packages through it.

#### Example: Installing NumPy

Using `pip`:
```sh
pip install numpy
```

Using `uv`:
```sh
uv add numpy
```

NumPy is a widely-used library in Python for numerical computations and is often required in various projects.

### Writing and Running a Simple Program

1. Open a text editor and write the following code:
    ```python
    print("Hello, World!")
    ```
2. Save the file as `hello.py`.
3. Open a terminal (macOS) or Command Prompt (Windows) and navigate to the directory containing your file.
4. Run the program:
    ```sh
    python hello.py
    ```
   If you’re using UV, and the project has been initialized, you can run the program with:
   ```sh
   uv run hello.py
   ```

### Getting started with this project
The code is structure like this, we have a main.py file, where the machine learning code is written. The goal will be to complete the tests written in test_public.py. To run the project, start by installing the packages. 

If you use pip, run this command
```sh
pip install -r requirements.txt
```
this will install all the libraries.

For uv run
```sh
uv sync
```

For å kjøre testene, kjør

```sh
pytest
```
eller
```sh
uv run pytest
```

### Introduction to Notebooks

Notebooks, like Jupyter, provide an interactive environment for writing and running Python code. They’re particularly useful for data analysis, visualization, and prototyping.

#### Getting Started with Jupyter in VS Code

1. Install the Jupyter kernel for Python:
    ```sh
    pip install ipykernel
    ```
    Or, using UV:
    ```sh
    uv add ipykernel
    ```
2. Install the Jupyter extension in VS Code.
3. Reload the VS Code window.
4. Create a new file named `notebook.ipynb` and open it.
5. Add code to the first cell (use `+Code` if a cell isn’t automatically created).
6. Run the cell by clicking the play button or pressing `Shift + Enter`.

Jupyter notebooks allow you to execute code cell-by-cell, making it easy to test and debug individual sections of your script. This functionality leverages Python’s interpreted nature, executing code line by line and maintaining the state across cells.

---

This guide provides the essential steps to start your Python journey. With Python installed and a basic understanding of tools like UV and Jupyter, you're ready to explore its capabilities further!

