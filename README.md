This is a Disc Recommender application written in Python for my WGU Capstone Project.

This application takes in a user input of a disc golf disc model and outputs 10 comparable discs using Cosine Similarity based on the discs flight numbers and disc type.

The intial dataset for this project can be found at https://www.kaggle.com/datasets/jakestrasler/disc-golf-disc-flight-numbers-and-dimensions?resource=download

User Guide
1.	In your web browser go to https://mybinder.org/
2.	Paste this URL https://github.com/J-Klimek/DiscRecommender in the box for ‘GitHub repository name or URL’
3.	Click the orange launch button and wait for the JupyterLab tab to open.
4.	Double click on the discRecommender.ipynb file in the left file navigation panel.
  a.	The necessary packages are set to install automatically when the file is opened however if you get an error click on the Run drop down in the menu bar and choose Run all cells. This should redownload the necessary packages and run the application.
5.	Once the application has run you may need to scroll down below the source code for the UI.
6.	There will be a prompt “Enter a disc by model name:”
  a.	Input a model and press Enter.
  b.	Sample models to use if unknown:
    i.	Buzzz
  ii.	Aviar
  iii.	D1
  iv.	Destroyer
    v.	M4 
7.	The Disc Recommender will output the top 10 recommended discs based on the discs flight attributes and disc type (Putt/Approach, Mid Range, Control Driver or Distance Driver)
8.	The prompt will ask the user “Compare another disc?”
a.	Type y to be prompted for another disc model
b.	Type n to finish the application.
9.	Upon typing N, the application will output 2 graphs explaining how the recommender system works using a comparison algorithm. The first graph is a histogram showing the similarity scale and the second is a heatmap showing 10 random discs and how similar they are compared to other discs in the dataset using a colored scale.
