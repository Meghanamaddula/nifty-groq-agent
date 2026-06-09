import os
import json
import yfinance as yf
import pandas as pd
from groq import Groq
from dotenv import load_dotenv


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


NIFTY_50_SECTORS = {
    "RELIANCE.NS": ("Reliance Industries", "Oil, Gas & Consumable Fuels"),
    "TCS.NS": ("Tata Consultancy Services", "Information Technology"),
    "HDFCBANK.NS": ("HDFC Bank", "Financial Services"),
    "ICICIBANK.NS": ("ICICI Bank", "Financial Services"),
    "INFY.NS": ("Infosys", "Information Technology"),
    "SBIN.NS": ("State Bank of India", "Financial Services"),
    "BHARTIARTL.NS": ("Bharti Airtel", "Telecommunication"),
    "ITC.NS": ("ITC", "FMCG"),
    "LT.NS": ("Larsen & Toubro", "Construction"),
    "AXISBANK.NS": ("Axis Bank", "Financial Services"),
    "KOTAKBANK.NS": ("Kotak Mahindra Bank", "Financial Services"),
    "HINDUNILVR.NS": ("Hindustan Unilever", "FMCG"),
    "BAJFINANCE.NS": ("Bajaj Finance", "Financial Services"),
    "ASIANPAINT.NS": ("Asian Paints", "Consumer Durables"),
    "MARUTI.NS": ("Maruti Suzuki", "Automobile"),
    "SUNPHARMA.NS": ("Sun Pharma", "Healthcare"),
    "TITAN.NS": ("Titan", "Consumer Durables"),
    "ULTRACEMCO.NS": ("UltraTech Cement", "Construction Materials"),
    "WIPRO.NS": ("Wipro", "Information Technology"),
    "ONGC.NS": ("ONGC", "Oil, Gas & Consumable Fuels"),
    "NTPC.NS": ("NTPC", "Power"),
    "POWERGRID.NS": ("Power Grid", "Power"),
    "TATASTEEL.NS": ("Tata Steel", "Metals"),
    "JSWSTEEL.NS": ("JSW Steel", "Metals"),
    "COALINDIA.NS": ("Coal India", "Oil, Gas & Consumable Fuels"),
    "HCLTECH.NS": ("HCL Technologies", "Information Technology"),
    "TECHM.NS": ("Tech Mahindra", "Information Technology"),
    "BAJAJFINSV.NS": ("Bajaj Finserv", "Financial Services"),
    "ADANIENT.NS": ("Adani Enterprises", "Metals"),
    "ADANIPORTS.NS": ("Adani Ports", "Services"),
    "GRASIM.NS": ("Grasim Industries", "Construction Materials"),
    "HINDALCO.NS": ("Hindalco", "Metals"),
    "M&M.NS": ("Mahindra & Mahindra", "Automobile"),
    "TMPV.NS": ("Tata Motors Passenger Vehicles", "Automobile"),
    "HEROMOTOCO.NS": ("Hero MotoCorp", "Automobile"),
    "EICHERMOT.NS": ("Eicher Motors", "Automobile"),
    "NESTLEIND.NS": ("Nestle India", "FMCG"),
    "BRITANNIA.NS": ("Britannia", "FMCG"),
    "CIPLA.NS": ("Cipla", "Healthcare"),
    "DRREDDY.NS": ("Dr Reddy's Laboratories", "Healthcare"),
    "APOLLOHOSP.NS": ("Apollo Hospitals", "Healthcare"),
    "DIVISLAB.NS": ("Divi's Laboratories", "Healthcare"),
    "BPCL.NS": ("BPCL", "Oil, Gas & Consumable Fuels"),
    "IOC.NS": ("Indian Oil Corporation", "Oil, Gas & Consumable Fuels"),
    "TATACONSUM.NS": ("Tata Consumer Products", "FMCG"),
    "BAJAJ-AUTO.NS": ("Bajaj Auto", "Automobile"),
    "INDUSINDBK.NS": ("IndusInd Bank", "Financial Services"),
    "SHRIRAMFIN.NS": ("Shriram Finance", "Financial Services"),
    "SBILIFE.NS": ("SBI Life Insurance", "Financial Services"),
    "HDFCLIFE.NS": ("HDFC Life Insurance", "Financial Services"),
}


def get_sector_performance():
    tickers = list(NIFTY_50_SECTORS.keys())

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

            if len(stock_data) < 2:
                continue

            previous_close = stock_data.iloc[-2]["Close"]
            latest_close = stock_data.iloc[-1]["Close"]

            percent_change = ((latest_close - previous_close) / previous_close) * 100

            company, sector = NIFTY_50_SECTORS[ticker]

            rows.append({
                "ticker": ticker,
                "company": company,
                "sector": sector,
                "latest_close": round(float(latest_close), 2),
                "previous_close": round(float(previous_close), 2),
                "percent_change": round(float(percent_change), 2),
            })

        except Exception:
            continue

    df = pd.DataFrame(rows)

    sector_df = (
        df.groupby("sector")
        .agg(
            average_return_percent=("percent_change", "mean"),
            number_of_companies=("company", "count"),
        )
        .reset_index()
        .sort_values(by="average_return_percent", ascending=False)
    )

    return sector_df.to_dict(orient="records")


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_sector_performance",
            "description": "Calculate sector-wise average daily performance for Nifty 50 stocks.",
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
                "You are a stock market analyst assistant. "
                "Use the tool result only. Explain sector performance in simple beginner-friendly language."
            ),
        },
        {
            "role": "user",
            "content": "Fetch sector-wise performance for Nifty 50 and explain strongest and weakest sectors.",
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
            if tool_call.function.name == "get_sector_performance":
                tool_result = get_sector_performance()

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