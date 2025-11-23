# Stock Predictor & Insight: Google, Apple & Microsoft

A Streamlit web application that predicts stock prices for Google (GOOGL), Apple (AAPL), and Microsoft (MSFT) using a trained machine learning model and provides AI-generated explanations or critiques of the predictions.

---

## Features

- **Natural Language Input**: Users can describe stock movements in plain English.
- **Automatic Data Extraction**: Converts user sentences into structured stock data (Open, High, Low, Volume, Ticker, Date) using an LLM (Groq LLaMA 3.1).
- **Prediction**: Uses a pre-trained machine learning model to predict stock prices.
- **Explanation / Critique**: Provides an AI-generated reasoning for the predicted stock price.
- **Interactive Chat Interface**: Users can enter multiple queries in a chat-like interface with session history.

---

## Technologies Used

- [Streamlit](https://streamlit.io/) – For the web interface.
- [LangChain](https://www.langchain.com/) – For LLM prompt chaining.
- [Groq LLaMA 3.1](https://www.groq.com/) – For natural language understanding and JSON extraction.
- [Scikit-learn](https://scikit-learn.org/) – For machine learning model prediction.
- [Pandas](https://pandas.pydata.org/) – For data handling.
- [Pickle](https://docs.python.org/3/library/pickle.html) – For loading pre-trained models and encoders.
- [Python Dotenv](https://pypi.org/project/python-dotenv/) – For managing API keys.

---

## Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd <repository-directory>
Install dependencies

bash
Copy code
pip install -r requirements.txt
Set up API Key

Create a .env file with your Groq API key:

ini
Copy code
GROQ_API_KEY=your_api_key_here
Usage
Run the Streamlit app:

bash
Copy code
streamlit run main.py
Enter your Groq API key in the sidebar if not already set.

Type a natural-language description of the stock movement in the chat box, for example:

csharp
Copy code
Google shares went up today, closing at 171 with a trading volume of 2.3 million.
The app will display:

Predicted stock value.

AI-generated analysis or critique explaining the prediction.

Session history of user queries and predictions.

Folder Structure
bash
Copy code
.
├── main.py               # Streamlit app
├── Training/
│   ├── Models/
│   │   └── model.pkl     # Pre-trained ML model
│   ├── Scaler/
│   │   └── scaler.pkl    # Pre-fitted scaler
│   └── Encoders/
│       └── encoder.pkl   # LabelEncoder / OneHotEncoder for ticker
├── .env                  # Environment file for API keys
├── requirements.txt      # Python dependencies
└── README.md
```
## Notes

- The ML model expects structured numeric data. Any missing fields are inferred by the LLM and filled before prediction.
- The second LLM call provides a textual critique or reasoning for transparency and interpretability.
- Currently supports only **GOOGL**, **AAPL**, and **MSFT** tickers.

## License

This project is licensed under the MIT License.
