<b>DeepBack</b>, a predictive model capable of determining corrected mass shifts in hydrogen-deuterium exchange mass spectrometry (HDX-MS) profiles, was developed using a deep learning technique. To construct an input-output black-box mapping, the model was trained using in-house experimentally corrected HDX-MS data. <br /><br />
What is the significance of utilising this code? <br />
<br /> It is crucial to use corrected HDX-MS data when generating input for HDXmodeller, a modelling programme available on HDXsite (https://hdxsite.nms.kcl.ac.uk/). <br /> <br /> Python 3 or above, as well as the Python libraries Scikit-learn, NumPy, Scipy, Keras, and TensorFlow, are required to run the code. 
<br> Please report any bugs or issues in the issues section.<br />

Using the requirements.txt File

To ensure that you have all the necessary dependencies installed before running the code, we provide a requirements.txt file that lists the required Python packages along with their versions. Follow the steps below to set up the environment using this file:
1. Create a Virtual Environment (Optional but Recommended)

It's recommended to create a virtual environment to isolate the project's dependencies from your system-wide Python packages. If you're not familiar with virtual environments, you can create one as follows:
<pre>python -m venv myenv
source myenv/bin/activate      # On Windows, use: myenv\Scripts\activate
</pre>



 2. Install Dependencies

Once you have your virtual environment activated (if you're using one), navigate to the project directory and install the dependencies using the pip command:

<pre>pip install -r requirements.txt
</pre>



This command reads the requirements.txt file and installs the specified packages along with their versions.

 3. Verifying Installation

After the installation is complete, you can verify that the required packages are installed in your environment by using the following command:

<pre>pip list
</pre>


This will display a list of installed packages along with their versions.
 4. Running the Code

With the dependencies installed, you can now run your code as usual. For example, you can execute your script using:

<pre>
  python DeepBack.py
</pre>


