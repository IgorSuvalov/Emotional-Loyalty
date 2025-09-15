import pandas
import re


def clean_data(df):

    df = df.drop(columns=['Item Purchased', 'Location', 'Size', 'Color', 'Shipping Type', 'Payment Method',
                          'Preferred Payment Method', 'Age', 'Gender', 'Category',
                          'Season', 'Promo Code Used', 'Previous Purchases'])

    # Write the columns with numerical and categorical entries out
    categorical = ['Subscription Status', 'Discount Applied', 'Frequency of Purchases']

    # Standardise the names
    for cat in categorical:
        vals = (df[cat].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True).str.lower().unique())
        sorted(vals)

    # Apply to the dataset
    df['Subscription Status'] = yn_to01(df['Subscription Status'])
    df['Discount Applied'] = yn_to01(df['Discount Applied'])

    # Turn strings like 'Monthly' into events per year
    freq_map = {
        "daily": 365, "weekly": 52, "fortnightly": 26, "bi-weekly": 26,
        "monthly": 12, "every 3 months": 4, "quarterly": 4, "annually": 1,
    }
    df["Frequency of Purchases"] = df["Frequency of Purchases"].astype(str).str.lower().map(freq_map).fillna(1.0)

    df.columns = [to_snake(c) for c in df.columns]

    return df


# Map yes/no to 0/1
def yn_to01(s):
    m = s.astype(str).str.strip().str.lower()
    return m.map({"yes": 1, "no": 0}).astype(float)


# Make the column names consistent
def to_snake(name):
    name = re.sub(r"\W+", "_", str(name).strip())
    name = re.sub(r"__+", "_", name)
    return name.strip("_").lower()