echo
# SHOPSMARTAIDISCOUNTFINDER
📌 Features
🔍 Product Input – Enter any product name to analyze its price pattern

📈 Price Trend Visualization – Shows 30-day historical price chart

🤖 AI Price Prediction – Uses Linear Regression to forecast future prices

🧠 Smart Recommendation – Tells you whether to buy now or wait

🌐 Built with Streamlit for rapid web app deployment

🧠 Tech Stack
Category	Tools/Frameworks
Language	Python
ML Model	Linear Regression (scikit-learn)
UI Framework	Streamlit
Data Viz	Matplotlib, Pandas
Other	NumPy, datetime

🚀 Installation
Clone the repo and install dependencies:

bash
Copy
Edit
git clone https://github.com/your-username/shopsmart-discount-finder.git
cd shopsmart-discount-finder
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
📊 Example Screenshot
(You can insert a screenshot here if hosted on GitHub)

💡 How It Works
The app simulates 30 days of product price data

It trains a Linear Regression model to predict the price 7 days from now

It compares the predicted price with the average of the past prices

Based on this, it recommends Buy Now or Wait
