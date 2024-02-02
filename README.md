## Wolts Data Science Internship

##### ***For an interactive Colab version, click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1W9Pfu9UfNymQP5SKb2uO1jC0vCTzr390?usp=sharing)***

### Requirements

To ensure a seamless project execution without any environment conflicts, please ensure you have Conda installed:

1. **Install Conda:** If Conda is not installed, refer to the [Conda Installation Guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

### Commands

In the project's root, run the following command for assistance:

```bash
make
```

This will display the available commands:

- `clean`: Delete all compiled Python files
- `create_environment`: Set up the Python interpreter environment
- `data`: Perform data analysis
- `requirements`: Install Python dependencies
- `setup`: Run the setup script
- `test_environment`: Test if the Python environment is set up correctly

#### Step 1: Create Conda Environment

To create a Conda environment, execute the following command:

```bash
make create_environment
```

After creating the environment, follow the prompted instructions to activate the Conda environment.

#### Step 2: Install Requirements

To install project requirements within the Conda environment, run:

```bash
make requirements
```

#### Step 3: Run Analysis

To conduct data analysis, execute:

```bash
make data
```

Logs and results will be saved in the 'log' and 'result' folders.

*** Contact: ali.kaya at abo.fi ***