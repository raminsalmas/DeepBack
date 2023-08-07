<b>DeepBack</b>, a predictive model capable of determining corrected mass shifts in hydrogen-deuterium exchange mass spectrometry (HDX-MS) profiles, was developed using a deep learning technique. To construct an input-output black-box mapping, the model was trained using in-house experimentally corrected HDX-MS data. <br /><br />
What is the significance of utilising this code? <br />
<br /> It is crucial to use corrected HDX-MS data when generating input for HDXmodeller, a modelling programme available on HDXsite (https://hdxsite.nms.kcl.ac.uk/). <br /> <br /> Python 3 or above, as well as the Python libraries Scikit-learn, NumPy, Scipy, Keras, and TensorFlow, are required to run the code. 
<br> Please report any bugs or issues in the issues section.<br />

Installing Dependencies

Before you can run the neural network model for data analysis, you need to ensure that the required dependencies are installed on your system. These dependencies provide essential functionalities such as numerical operations, data visualization, machine learning, and more.

To install the necessary dependencies, follow the steps below:
Step 1: Install Python

Ensure you have Python installed on your system. If not, you can download and install it from the official Python website. This project is developed using Python 3.x.
Step 2: Install Required Packages

Open your command-line interface (terminal) and execute the following command to install the required packages using the pip package manager:

bash

pip install numpy matplotlib tensorflow scikit-learn scipy

Here's a brief explanation of each package:

    numpy: This package provides support for numerical operations and array manipulation.
    matplotlib: Matplotlib is a plotting library used for creating visualizations, graphs, and charts.
    tensorflow: TensorFlow is an open-source machine learning framework developed by Google. It's used here for building and training neural networks.
    scikit-learn: Scikit-learn is a machine learning library that provides tools for data preprocessing, modeling, and evaluation.
    scipy: SciPy is a library used for scientific and technical computing. It includes functions for optimization, interpolation, statistics, and more.

Step 3: Verify Installation

After installing the dependencies, you can verify their installation by executing the following command in your terminal:

bash

pip show numpy matplotlib tensorflow scikit-learn scipy

This command should display information about the installed packages, confirming that they are available for use.

