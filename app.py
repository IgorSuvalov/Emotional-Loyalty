from src.scoring import run_loyalty_model

import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title='Loyalty Model Interactive Demo', page_icon="shark", layout="wide")

st.title("Interactive Loyalty Model Demo", )
st.markdown("[GitHub page](https://github.com/IgorSuvalov/Emotional-Loyalty)")
st.divider()

@st.cache_data
def load_data():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(app_dir, "customer-shopping-latest-trends-dataset", "shopping_trends.csv")
    df = pd.read_csv(csv_path)
    return df


left_col, mid_col, right_col = st.columns(3)

left_col.subheader("Confidence parameter and engage/spend proportion")

mid_col.subheader("Final score boosts/penalties based on the archetype")

right_col.subheader("The tier distributions")

lam = left_col.slider('Confidence parameter', min_value=0.0, max_value=1.0, step=0.01, value=0.80)
left_col.write('Recommended range: 0.70 - 0.90')


def update(change):
    if change == 'Spend proportion':
        st.session_state['Engage proportion'] = round(1.0 - st.session_state['Spend proportion'], 2)
    else:
        st.session_state['Spend proportion'] = round(1.0 - st.session_state['Engage proportion'], 2)


spend = left_col.slider(
    'Spend proportion', value=0.60,step=0.05, key='Spend proportion',
    on_change=update, args=('Spend proportion',)
)
engage = left_col.slider(
    'Engage proportion', value=0.40, step=0.05, key='Engage proportion',
    on_change=update, args=('Engage proportion',)
)


Brand_ch = mid_col.slider('"Brand Champions"', min_value=-20, max_value=20, value=5, step=1, format="%d%%")
Trans_spend = mid_col.slider('"Transactional Spenders"', min_value=-20, max_value=20, value=-5, step=1, format="%d%%")
Brand_adv = mid_col.slider('"Brand Advocates"', min_value=-20, max_value=20, value=10, step=1, format="%d%%")
Pass_cons = mid_col.slider('"Passive Customers"', min_value=-20, max_value=20, value=-10, step=1, format="%d%%")

multipliers = {
    'Brand Champions':        1.00 + Brand_ch/100,
    'Transactional Spenders': 1.00 + Trans_spend/100,
    'Brand Advocates':       1.00 + Brand_adv/100,
    'Passive Customers':      1.00 + Pass_cons/100
}

mult_vals = multipliers.values()

Plat = right_col.number_input('Platinum (%)', min_value=0, max_value=100, value=10, step=1)/100
Gold = right_col.number_input('Gold (%)', min_value=0, max_value=100, value=15, step=1)/100
Silv = right_col.number_input('Silver (%)', min_value=0, max_value=100, value=25, step=1)/100
Reg = right_col.number_input('Regular (%)', min_value=0, max_value=100, value=50, step=1)/100

if Plat + Gold + Silv + Reg != 1.00:
    st.error("The tier distinction has to sum to 100%")
    tier_err = 1
else:
    tier_err = 0


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
    if tier_err != 0: st.warning("Make sure the values are in the correct ranges first!")

    else:
        final_df, block_sc, fig = run_loyalty_model(
            load_data(), knobs, lambda_parameter, multipliers, tier_mix
        )

        sample_df = final_df.sample(50, random_state=42)

        sample_df = sample_df.sort_values(by='score', ascending=False)

        sample_df = sample_df.drop(columns=["customer_id"])

        st.dataframe(sample_df, use_container_width=True, selection_mode="multi-row")
