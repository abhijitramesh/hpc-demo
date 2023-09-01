# Running Pytorch QuickStart code on HPC

## Credits
This code is adapted from [Pytorch Quick Start Guide](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)

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

## Folder Structure and File Descriptions

- `.gitignore`: Specifies files and directories that should be ignored by Git.
- `requirements.txt`: Lists the Python packages that need to be installed for the code to run.
- `main.py`: The main script that runs the model training and prediction.
- `utils/`: This directory contains utility scripts and modules that are used by `main.py`. Each file in this directory serves a specific purpose, such as data preprocessing, model definition, etc.

## Note

The model weights are saved in a `.pth` file (ignored by Git as specified in `.gitignore`). If you want to load a pre-trained model, make sure to place the `.pth` file in the correct directory.


