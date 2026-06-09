import os
import json
import yfinance as yf
import pandas as pd
from groq import Groq
from dotenv import load_dotenv


load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


NIFTY_50 = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "INFY.NS": "Infosys",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ITC.NS": "ITC",
    "LT.NS": "Larsen & Toubro",
    "AXISBANK.NS": "Axis Bank",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "BAJFINANCE.NS": "Bajaj Finance",
    "ASIANPAINT.NS": "Asian Paints",
    "MARUTI.NS": "Maruti Suzuki",
    "SUNPHARMA.NS": "Sun Pharma",
    "TITAN.NS": "Titan",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "WIPRO.NS": "Wipro",
    "ONGC.NS": "ONGC",
    "NTPC.NS": "NTPC",
    "POWERGRID.NS": "Power Grid",
    "TATASTEEL.NS": "Tata Steel",
    "JSWSTEEL.NS": "JSW Steel",
    "COALINDIA.NS": "Coal India",
    "HCLTECH.NS": "HCL Technologies",
    "TECHM.NS": "Tech Mahindra",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports",
    "GRASIM.NS": "Grasim Industries",
    "HINDALCO.NS": "Hindalco",
    "M&M.NS": "Mahindra & Mahindra",
    "TMPV.NS": "Tata Motors Passenger Vehicles",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "EICHERMOT.NS": "Eicher Motors",
    "NESTLEIND.NS": "Nestle India",
    "BRITANNIA.NS": "Britannia",
    "CIPLA.NS": "Cipla",
    "DRREDDY.NS": "Dr Reddy's Laboratories",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "DIVISLAB.NS": "Divi's Laboratories",
    "BPCL.NS": "BPCL",
    "IOC.NS": "Indian Oil Corporation",
    "TATACONSUM.NS": "Tata Consumer Products",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "INDUSINDBK.NS": "IndusInd Bank",
    "SHRIRAMFIN.NS": "Shriram Finance",
    "SBILIFE.NS": "SBI Life Insurance",
    "HDFCLIFE.NS": "HDFC Life Insurance",
}


def get_top_5_nifty_by_volume():
    tickers = list(NIFTY_50.keys())

    data = yf.download(
        tickers=tickers,
        period="10d",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    rows = []

    for ticker in tickers:
        try:
            stock_data = data[ticker].dropna()
            stock_data = stock_data[stock_data["Volume"] > 0]

            if stock_data.empty:
                continue

            latest_row = stock_data.iloc[-1]
            latest_date = str(stock_data.index[-1].date())

            rows.append({
                "ticker": ticker,
                "company": NIFTY_50[ticker],
                "trading_date": latest_date,
                "close_price": round(float(latest_row["Close"]), 2),
                "volume": int(latest_row["Volume"]),
            })

        except Exception:
            continue

    result = sorted(rows, key=lambda x: x["volume"], reverse=True)[:5]
    return result


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_top_5_nifty_by_volume",
            "description": "Fetch top 5 Nifty 50 companies sorted by latest completed trading day's volume.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
]


def run_agent():
    messages = [
        {
            "role": "system",
            "content": (
                "You are a stock market data assistant. "
                "Use the provided tool to fetch real stock data. "
                "Do not invent numbers. Explain results simply."
            ),
        },
        {
            "role": "user",
            "content": "Find the top 5 Nifty 50 companies by traded volume for the latest completed trading day.",
        },
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    assistant_message = response.choices[0].message
    messages.append(assistant_message)

    if assistant_message.tool_calls:
        for tool_call in assistant_message.tool_calls:
            if tool_call.function.name == "get_top_5_nifty_by_volume":
                tool_result = get_top_5_nifty_by_volume()

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": json.dumps(tool_result),
                })

        final_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
        )

        print(final_response.choices[0].message.content)

    else:
        print(assistant_message.content)


if __name__ == "__main__":
    run_agent()