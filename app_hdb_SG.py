import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the pre-trained XGBoost model and the encoder for categorical features
model = joblib.load("xgb_hdb_price_model.pkl")
encoder = joblib.load("ordinal_encoder.pkl")

# Configure the Streamlit app UI (title, icon, layout)
st.set_page_config(page_title="HDB Price Estimator", page_icon="🏠", layout="centered")

st.title("🏘️ Singapore HDB Resale Price Estimator")
st.markdown(
    "Estimate the resale price of an HDB flat in Singapore using a machine learning model "
    "trained on historical transactions and economic indicators."
)
st.divider()

# Dropdown options for user inputs
towns = ['Ang Mo Kio', 'Bedok', 'Bishan', 'Bukit Batok', 'Bukit Merah', 'Bukit Panjang',
         'Bukit Timah', 'Central Area', 'Choa Chu Kang', 'Clementi', 'Geylang', 'Hougang',
         'Jurong East', 'Jurong West', 'Kallang/Whampoa', 'Marine Parade', 'Pasir Ris',
         'Punggol', 'Queenstown', 'Sembawang', 'Sengkang', 'Serangoon', 'Tampines',
         'Toa Payoh', 'Woodlands', 'Yishun']

flat_types = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']
flat_models = ['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified', 'Premium Apartment',
               'Maisonette', 'Apartment', 'DBSS', 'Type S1', 'Type S2']

mrt_stations = ['Mayflower MRT Station', 'Ang Mo Kio MRT Station', 'Yio Chu Kang MRT Station',
                'Lentor MRT Station', 'Bedok Reservoir MRT Station', 'Kembangan MRT Station',
                'Kaki Bukit MRT Station', 'Tanah Merah MRT Station', 'Bedok North MRT Station',
                'Ubi MRT Station', 'Bishan MRT Station', 'Braddell MRT Station',
                'Upper Thomson MRT Station', 'Marymount MRT Station', 'Bukit Batok MRT Station',
                'Chinese Garden MRT Station', 'Bukit Gombak MRT Station', 'Tiong Bahru MRT Station',
                'Outram Park MRT Station', 'Queenstown MRT Station', 'HarbourFront MRT Station',
                'Redhill MRT Station', 'Labrador Park MRT Station', 'Telok Blangah MRT Station',
                'Bukit Panjang MRT Station', 'Cashew MRT Station', 'Beauty World MRT Station',
                'Farrer Road MRT Station', 'Bugis MRT Station', 'Jalan Besar MRT Station',
                'Chinatown MRT Station', 'Tanjong Pagar MRT Station', 'Little India MRT Station',
                'Choa Chu Kang MRT Station', 'Yew Tee MRT Station', 'Clementi MRT Station',
                'Dover MRT Station', 'Eunos MRT Station', 'Aljunied MRT Station',
                'Mountbatten MRT Station', 'Mattar MRT Station', 'Paya Lebar MRT Station',
                'MacPherson MRT Station', 'Geylang Bahru MRT Station', 'Dakota MRT Station',
                'Tai Seng MRT Station', 'Kovan MRT Station', 'Serangoon MRT Station',
                'Hougang MRT Station', 'Buangkok MRT Station', 'Jurong East MRT Station',
                'Lakeside MRT Station', 'Boon Lay MRT Station', 'Pioneer MRT Station',
                'Nicoll Highway MRT Station', 'Farrer Park MRT Station', 'Boon Keng MRT Station',
                'Toa Payoh MRT Station', 'Kallang MRT Station', 'Esplanade MRT Station',
                'Lavender MRT Station', 'Pasir Panjang MRT Station', 'Pasir Ris MRT Station',
                'Tampines East MRT Station', 'Buona Vista MRT Station', 'Commonwealth MRT Station',
                'Holland Village MRT Station', 'one-north MRT Station', 'Lorong Chuan MRT Station',
                'Simei MRT Station', 'Tampines MRT Station', 'Upper Changi MRT Station',
                'Bartley MRT Station', 'Potong Pasir MRT Station', 'Woodleigh MRT Station',
                'Marsiling MRT Station', 'Woodlands MRT Station', 'Woodlands North MRT Station',
                'Admiralty MRT Station', 'Yishun MRT Station', 'Canberra MRT Station',
                'Khatib MRT Station', 'Bras Basah MRT Station', 'Rochor MRT Station',
                'Botanic Gardens MRT Station', 'Bencoolen MRT Station', 'Clarke Quay MRT Station',
                'King Albert Park MRT Station', 'Novena MRT Station', 'Bendemeer MRT Station',
                'Changi Airport MRT Station', 'Woodlands South MRT Station', 'Sembawang MRT Station',
                'Sengkang MRT Station', 'Caldecott MRT Station', 'Punggol MRT Station']

# Storey ranges mapped to their median floor (helps simplify floor inputs)
storey_ranges = {
    '01 TO 03': 2, '04 TO 06': 5, '07 TO 09': 8, '10 TO 12': 11,
    '13 TO 15': 14, '16 TO 18': 17, '19 TO 21': 20, '22 TO 24': 23,
    '25 TO 27': 26, '28 TO 30': 29, '31 TO 33': 32, '34 TO 36': 35,
    '37 TO 39': 38, '40 TO 42': 41, '43 TO 45': 44, '46 TO 48': 47,
    '49 TO 51': 50
}

# Define function to convert a floor number into the median floor of its range
def map_floor_to_median(floor_input):
    for rng, median in storey_ranges.items():
        low, high = map(int, rng.split(" TO "))
        if low <= floor_input <= high:
            return median
    return np.nan

# Main input form for users to select property details
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        town = st.selectbox("Town", towns)
        flat_type = st.selectbox("Flat Type", flat_types)
        floor_area_sqm = st.slider("Floor Area (sqm)", 30, 200, 80)
        floor_input = st.slider("Storey", 1, 51, 5)
    with col2:
        flat_model = st.selectbox("Flat Model", flat_models)
        closest_mrt = st.selectbox("Closest MRT Station", mrt_stations)
        cbd_dist = st.slider("Distance to CBD (km)", 0, 25, 8)
        lease_year = st.number_input("Lease Commencement Year", 1960, 2025, 2000)

    submitted = st.form_submit_button("🔍 Estimate Price")

# Run prediction once the user clicks submit
if submitted:
    storey_median = map_floor_to_median(int(floor_input))

    if np.isnan(storey_median):
        st.error("Invalid floor number. Must be between 1 and 51.")
    else:
        cbd_dist_meters = cbd_dist * 1000
        # Collect all user inputs into a single DataFrame
        user_df = pd.DataFrame([{
            'floor_area_sqm': floor_area_sqm,
            'storey_median': storey_median,
            'closest_mrt': closest_mrt,
            'cbd_dist': cbd_dist_meters,
            'lease_commence_date': lease_year,
            'town': town,
            'flat_model': flat_model,
            'flat_type': flat_type,
            # Fixed background economic indicators
            'GDP': 510000.0,
            'GDP per Capita': 85000.0,
            'GNI per Capita': 79000.0,
            'Population': 5600000,
            'Resident Household': 1200000,
            'HDB Dwellings': 1000000,
            'Inflation Rate': 1.5,
            'Interest Rate': 3.0,
            'Unemployment Rate': 2.1,
            'Yield_5y_interest': 2.5
        }])

        # Apply the encoder to convert categorical inputs into numeric form
        user_df[['town', 'flat_model', 'flat_type', 'closest_mrt']] = encoder.transform(
            user_df[['town', 'flat_model', 'flat_type', 'closest_mrt']]
        )

        # Make prediction with the trained model
        prediction = model.predict(user_df)[0]

        # Display the result nicely formatted
        st.success(f"💰 **Estimated Resale Price:** SGD {prediction:,.2f}")

        st.info("ℹ️ Note: This estimate is based on historical and economic data. "
                "Actual resale prices may vary depending on market conditions.")
