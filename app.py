from src.scoring import run_loyalty_model

import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title='Loyalty Model Interactive Demo', page_icon="shark", layout="wide")

st.title("Interactive Loyalty Model Demo", )
st.divider()

@st.cache_data
def load_data():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(app_dir, "customer-shopping-latest-trends-dataset", "shopping_trends.csv")
    df = pd.read_csv(csv_path)
    return df

left_col, mid_col, right_col = st.columns(3)

left_col.subheader("Confidence parameter and engage/spend proportion")

mid_col.subheader("The penalties and boosts for the archetypes")

right_col.subheader("The tier distributions")

lam = left_col.slider('Confidence parameter', min_value=0.0, max_value=1.0, step=0.01, value=0.80)
left_col.write('Recommended range: 0.70 - 0.90')

spend = left_col.number_input('Spend proportion', value=0.60,step=0.05)
engage = left_col.number_input('Engage proportion', value=0.40, step=0.05)

if spend + engage != 1.0:
    st.error('Make the spend and engage sum to 1.0!')

Brand_ch = mid_col.number_input('"Brand Champions"', value=1.10, step=0.01)
Trans_spend = mid_col.number_input('"Transactional Spenders"', value=0.95, step=0.01)
Brand_adv = mid_col.number_input('"Brand Advocates"', value=1.15, step=0.01)
Pass_cons = mid_col.number_input('"Passive Customers"', value=0.90, step=0.01)

multipliers = {
    'Brand Champions':        Brand_ch,
    'Transactional Spenders': Trans_spend,
    'Brand Advocates':       Brand_adv,
    'Passive Customers':      Pass_cons
}

mult_vals = multipliers.values()
if min(mult_vals) < 0.80 or max(mult_vals) > 1.20:
    st.error('The maximum boost/penalty is 20%!')

Plat = right_col.number_input('Platinum', value=0.10, step=0.05)
Gold = right_col.number_input('Gold', value=0.15, step=0.05)
Silv = right_col.number_input('Silver', value=0.25, step=0.05)
Reg = right_col.number_input('Regular', value=0.50, step=0.05)

if Plat + Gold + Silv + Reg != 1.00:
    st.error("The tier distinction has to sum to 100%")

knobs = {
    "spend":  spend,
    "engage": engage,
}
lambda_parameter = lam

tier_mix = {
    "Platinum": Plat,
    "Gold": Gold,
    "Silver": Silv,
    "Regular": Reg
}

st.subheader("Model's output using parameters above", divider="green")

if st.button("Run the model"):
    final_df, block_sc, fig = run_loyalty_model(load_data(), knobs, lambda_parameter, multipliers, tier_mix)

    sample_df = final_df.sample(50, random_state=42)

    sample_df = sample_df.sort_values(by='score', ascending=False)

    sample_df = sample_df.drop(columns=["customer_id"])

    st.dataframe(sample_df, use_container_width=True)

