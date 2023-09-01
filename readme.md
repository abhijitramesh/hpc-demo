# Running Pytorch Quickstart code on HPC

## Credits
This code is adapted from [Pytorch Quickstart Guide](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)

## Running Locally

1. Clone the repository to your local machine.
2. Create a virtual environment:
   ```
   python -m venv env
   ```
3. Activate the virtual environment:
   - On Windows:
     ```
     .\env\Scripts\activate
     ```
   - On Unix or MacOS:
     ```
     source env/bin/activate
     ```
4. Install the required packages using pip:
   ```
   pip install -r requirements.txt
   ```
5. Run the main script:
   ```
   python main.py
   ```

## Running on HPC

1. Download and unzip the repository to your home directory on the Altair access website (e.g., `hpccluster/home/username`).

2. Start a Jupyter notebook with CPU via the jobs tab.

3. Open a terminal session in the Jupyter notebook and navigate to the unzipped code.

4. Set up a virtual environment and activate it:
   ```
   python -m venv env
   source env/bin/activate
   ```

5. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

6. Update the `which_python` and `main_path` variables in `run_on_hpc.py`:
   - Run `which python` to get the path of your virtual environment's Python (e.g., `~/hpc-demo-main/env/bin/python`). Replace `~` with `/home/username`.
   - Update `which_python`:
   ```python
   which_python = '/home/username/hpc-demo-main/env/bin/python'
   ```
   - Update `main_path` with the path to `main.py`:
   ```python
   main_path = "/home/username/hpc-demo-main/main.py"
   ```

7. Start a job with GPU:
   - Go to the jobs tab and select Shell Script.
   - Check the GPU option.
   - Provide `run_on_hpc.py` as your job script.

    
## Folder Structure and File Descriptions

- `.gitignore`: Specifies files and directories that should be ignored by Git.
- `requirements.txt`: Lists the Python packages that need to be installed for the code to run.
- `main.py`: The main script that runs the model training and prediction.
- `utils/`: This directory contains utility scripts and modules that are used by `main.py`. Each file in this directory serves a specific purpose, such as data preprocessing, model definition, etc.
- `run_on_hpc.py`: The script to run main.py using a venv created on the hpc
## Note

The model weights are saved in a `.pth` file (ignored by Git as specified in `.gitignore`). If you want to load a pre-trained model, make sure to place the `.pth` file in the correct directory.
