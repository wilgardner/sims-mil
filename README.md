This repo contains Python scripts and an example notebook relating to our paper:

[Markedly Enhanced Analysis of Mass Spectrometry Images Using Weakly Supervised Machine Learning](https://onlinelibrary.wiley.com/doi/10.1002/smtd.202301230)

This TensorFlow implementation provides a framework for applying our multiple instance learning (MIL) method to hyperspectral time-of-flight secondary ion mass spectrometry (ToF-SIMS) imaging data (or, in principle, any hyperspectral data). 
Designed for weakly supervised learning scenarios, this method is useful in situations when labels are available at the bag-level (or image-level), but not at the instance-level (or pixel-level).

# Citation
If you use this method in your research, please cite our [paper](https://onlinelibrary.wiley.com/doi/10.1002/smtd.202301230).

# Installation
## Requirements
- Python 3.6+
- TensorFlow 2.x
- NumPy
- SciPy
- Matplotlib
- Seaborn
# Setup
 
Clone this repository and install the required Python packages.

# Usage
1. **Data Preparation**: Organize your hyperspectral ToF-SIMS imaging data into a directory structure with separate subfolders for each binary class. Each file must be a separate data cube of size h X w X p. Currently, either '.mat' (MATLAB) or '.npy' (NumPy) file formats are supported - feel free to add your own import logic to the **utils.load_data_from_paths** function.

2. **Configuration**: Update the Parameters section in the provided Jupyter notebook to point to your data folder and adjust the training parameters as needed.

3. **Model Training**: Run the notebook to train the MIL model on your data. The notebook covers data loading, preprocessing, model fitting, and evaluation.

4. **Evaluation**: After training, the notebook provides utilities to evaluate the model's performance and visualize attention maps, which show the characteristic pixels of each binary class.

# Important
This repo is intended to give an example of (1) a basic implementation of the MIL method in our [paper](https://onlinelibrary.wiley.com/doi/10.1002/smtd.202301230) and (2) how to train the MIL model and employ inference on test data. It does NOT cover many critical aspects of model development, optimisation and utilisation, such as cross-validation, hyperparameter tuning, variations in data preprocessing, extraction of characteristic spectra etc. You are encouraged to use this repo as a *starting point* and modify/extend on it as needed for your own use case.

# Acknowledgments
This code was written by Wil Gardner, Centre for Materials and Surface Science (Prof. Paul Pigram, Director), La Trobe University. Special thanks to the contributors and researchers who have laid the groundwork in the fields of MIL and ToF-SIMS image analysis. In particular, thanks to the creators of the [dual-stream multiple instance learning (DSMIL)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8765709/) method, upon which our method is based.

# License
This project is licensed under the Attribution-NonCommercial-NoDerivatives 4.0 International License - see the LICENSE file for details.

