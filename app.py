import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Title
st.title("🛍️ ShopSmart - AI Discount Finder")
st.markdown("Get smart recommendations for the best time to buy your favorite products based on price trends!")

# User input
product_name = st.text_input("🔍 Enter Product Name:")

# Simulated price trend data (for demonstration)
def generate_dummy_data():
    dates = pd.date_range(end=datetime.today(), periods=30).to_pydatetime().tolist()
    prices = np.random.normal(loc=100, scale=10, size=30)
    return pd.DataFrame({"Date": dates, "Price": prices})

# Main logic
if product_name:
    st.subheader(f"📈 Price Trend for: **{product_name}**")

    # Simulate historical price data
    data = generate_dummy_data()
    data['Timestamp'] = data['Date'].map(datetime.timestamp)

    # Train simple linear regression model
    X = data['Timestamp'].values.reshape(-1, 1)
    y = data['Price'].values
    model = LinearRegression().fit(X, y)

    # Predict price for 7 days in the future
    future_date = datetime.today() + timedelta(days=7)
    future_timestamp = datetime.timestamp(future_date)
    predicted_price = model.predict([[future_timestamp]])[0]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data['Date'], data['Price'], label='Historical Prices', marker='o')
    ax.axvline(x=future_date, color='green', linestyle='--', label='Prediction Point')
    ax.scatter(future_date, predicted_price, color='red', label=f'Predicted Price: ₹{predicted_price:.2f}')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    ax.set_title("Price Trend")
    ax.legend()
    st.pyplot(fig)

    # Recommendation Logic
    if predicted_price < y.mean():
        rec = f"✅ Best time to buy in next 7 days. Expected price: ₹{predicted_price:.2f}"
    else:
        rec = f"⚠️ Consider waiting. Expected price in 7 days: ₹{predicted_price:.2f}"

    st.markdown("### 🧠 AI Recommendation")
    st.success(rec)

    st.markdown("#### 📝 How it works:")
    st.info(
        "We analyze 30 days of simulated price trends using linear regression and predict the expected price 7 days from today. "
        "If the expected price is lower than the average of the last 30 days, we recommend buying soon."
    )

# Footer
st.markdown("---")
st.caption("Developed by Md Kaunain | Streamlit + ML Regression Demo")
