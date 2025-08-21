# Emotional-Loyalty model with an interactive page

## A Jupyter notebook and an interactive Streamlit page for a loyalty scoring model accounting for the customer's engagement with the product. 

The idea of this project is to provide a customisable model for loyalty scoring that detects "archetypes" of customers via unsupervised learning and assigns scores/tiers based of the learned archetype and the engagement/spending contributions. The project consists of two main parts:

* Jupyter notebook going through the model's architecture.
  * Preprocessing of the data
  * Unsupervised learning (archetype detection) via K-means clustering
  * Function `U(x)` assigning scores based on the customer's engagement/spending. 
  * Tierer function combining the archetype and `U(x)` contributions in proportions based on the confidence parameter $\\lambda$. Allows for score penalties/boost for each archetype

* Interactive Streamlit page allowing to run the model on a subset of the data to see how altering the parameters changes the score/tier distribution. 

## Tech stack
* **Modeling:** Python, scikit-learn, Pandas, NumPy
* **Interactive app:** Streamlit
* **Data visualisation:** Matplotlib

## Streamlit page [(click here)](https://igorshuv-emotional-loyalty.streamlit.app/) 

will add a video here

## Parameters
* **Confidence $\\lambda$** - How much we trust the `U(x)` contribution vs the archetype contribution. 
* **Spend vs Engage** - how much we prioritise contributions to the score based of customer's spending habits vs product engagement (must add to 1.00).
* **Archetype multipliers** - penalties/boosts to the score based on the archetype of the customer. Allowed range 0.80-1.20 (+-20%)
* **Tier distribution** - proportions of Platinum/Gold/Silver/Regular (must add to 1.00)

## How to install this project

1.  Clone the project.
    ```bash
    git clone [https://github.com/IgorSuvalov/Emotional-Loyalty](https://github.com/IgorSuvalov/Emotional-Loyalty)
    cd Emotional-Loyalty
    ```
2.  Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the interactive app or use the link above for the streamlit page.
    ```bash
    streamlit run app.py
    ```
4.  Explore the analysis.
    Model analysis is available in `Notebooks/notebooks.ipynb`.

## Repo layout

```
Emotional-Loyalty/
├───app.py  #streamlit app
├───customer-shopping-latest-trends-dataset
│   └───shopping_trends.csv
├───Notebooks/
│   └───notebook.ipynb  #analysis
├───results/
│   └───demo_config.json
├───src/
│   ├───__init__.py
│   ├───preprocessing.py
│   └───scoring.py
├───requirements.txt  #dependencies 
└───README.md


```