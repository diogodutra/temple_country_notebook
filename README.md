![](https://th.bing.com/th/id/R2182dde2e2eda8f978afbf46f0b80c0c?rik=4HeCF3rfmwSCHw&riu=http%3a%2f%2fwww.css.ethz.ch%2fen%2fthink-tank%2fthemes%2fmediation-support-and-peace-promotion%2freligion-and-mediation%2frmc%2f_jcr_content%2fpar%2ffullwidthimage%2fimage.imageformat.lightbox.493741631.jpg&ehk=TS6d2aReC7GIQ5bYZ0oqPOPYE4nklyBQZcLoIP%2fT6dE%3d&risl=&pid=ImgRaw)

<p>&nbsp;</p>

# &#x2615; Welcome to "Country of the Temple" project

Now we can guess from which country a temple is based on its picture! For this purpose, a predictive model was created from [this dataset](https://drive.google.com/file/d/1ccqGu9r815WvgHAlG2CujzUPOEW_Pvo9/view?usp=sharing) and it is available here.


<p>&nbsp;</p>

# &#x1f4c8; Data Analysis and Predictive Model

The detailed walkthough about the dataset analysis (ie: samples plots, insights) and the predictive model creation (ie: neural network architecture, choice of metrics, training and performance) are detailed on [this notebook](https://git.toptal.com/screening/diogo-dutra-2/-/blob/master/notebook/temple_country_pytorch.ipynb).


<p>&nbsp;</p>

# &#128187; Standalone script

The predictive model to tell the country of origin is [available here](https://git.toptal.com/screening/diogo-dutra-2/-/blob/master/temple_country.py) to run as a standalone Python script over all images in a folder. It is assumed that the environment is properly setup with all necessary modules installed (Python, NumPy, Torch, Torchvision and Pandas). It is also necessary to download the model parameters [from here](https://git.toptal.com/screening/diogo-dutra-2/-/tree/master/model) to a local subfolder named `model`. Then, you can run it from the terminal:
```
python temple_country.py ./folder_images 
```
The result is in the `results.csv` file created at the same location of `temple_country.py` file.

<p>&nbsp;</p>

# 📬 Get in touch

Feel free to contact me at anytime should you need further information about this project or for any other Machine Learning and Data Scientist:
- &#128100; Personal Web: [diogodutra.github.io](https://diogodutra.github.io)
- ![](https://i.stack.imgur.com/gVE0j.png) LinkedIn: [linkedin.com/in/diogodutra]()