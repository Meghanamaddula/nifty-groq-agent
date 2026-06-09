import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Note:
# Tata Motors old symbol TATAMOTORS.NS stopped returning data after its symbol
# change. Use TMPV.NS for Tata Motors Passenger Vehicles.
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


def _get_ticker_frame(downloaded_data, ticker):
    """Return one ticker's DataFrame from yfinance multi-ticker output."""
    if isinstance(downloaded_data.columns, pd.MultiIndex):
        if ticker in downloaded_data.columns.get_level_values(0):
            return downloaded_data[ticker].dropna(how="all")
        return pd.DataFrame()

    return downloaded_data.dropna(how="all")


def _format_volume_rows(rows, top_n=5):
    rows = [row for row in rows if row["volume"] is not None and row["volume"] > 0]
    rows = sorted(rows, key=lambda row: row["volume"], reverse=True)
    return rows[:top_n]


def get_last_and_current_day_nifty_volumes():
    """
    Fetch two separate Nifty 50 volume views:
    1. Last completed trading day volume.
    2. Current trading day volume so far, if Yahoo Finance has today's data.
    """
    india_tz = ZoneInfo("Asia/Kolkata")
    now_ist = datetime.now(india_tz)
    today_ist = now_ist.date()
    tickers = list(NIFTY_50.keys())

    daily_data = yf.download(
        tickers=tickers,
        period="15d",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    last_day_rows = []
    current_day_rows_from_daily = []
    last_completed_dates = []

    for ticker in tickers:
        frame = _get_ticker_frame(daily_data, ticker)

        if frame.empty or "Volume" not in frame.columns:
            continue

        frame = frame.dropna(subset=["Volume"])
        frame = frame[frame["Volume"] > 0]

        if frame.empty:
            continue

        frame = frame.copy()
        frame["trading_date"] = frame.index.date

        previous_days = frame[frame["trading_date"] < today_ist]
        today_rows = frame[frame["trading_date"] == today_ist]

        if not previous_days.empty:
            last_row = previous_days.iloc[-1]
            last_date = str(previous_days.index[-1].date())
            last_completed_dates.append(last_date)

            last_day_rows.append(
                {
                    "ticker": ticker,
                    "company": NIFTY_50[ticker],
                    "trading_date": last_date,
                    "close_price": round(float(last_row["Close"]), 2),
                    "volume": int(last_row["Volume"]),
                }
            )

        if not today_rows.empty:
            today_row = today_rows.iloc[-1]
            current_day_rows_from_daily.append(
                {
                    "ticker": ticker,
                    "company": NIFTY_50[ticker],
                    "trading_date": str(today_rows.index[-1].date()),
                    "close_price": round(float(today_row["Close"]), 2),
                    "volume": int(today_row["Volume"]),
                    "source": "daily candle",
                }
            )

    intraday_data = yf.download(
        tickers=tickers,
        period="1d",
        interval="5m",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    current_day_rows = []
    intraday_dates = []

    for ticker in tickers:
        frame = _get_ticker_frame(intraday_data, ticker)

        if frame.empty or "Volume" not in frame.columns:
            continue

        frame = frame.dropna(subset=["Volume"])

        if frame.empty:
            continue

        if frame.index.tz is None:
            frame = frame.tz_localize("UTC").tz_convert(india_tz)
        else:
            frame = frame.tz_convert(india_tz)

        today_frame = frame[frame.index.date == today_ist]

        if today_frame.empty:
            continue

        volume = int(today_frame["Volume"].sum())
        latest_close = today_frame["Close"].dropna()

        if volume <= 0 or latest_close.empty:
            continue

        intraday_dates.append(str(today_ist))

        current_day_rows.append(
            {
                "ticker": ticker,
                "company": NIFTY_50[ticker],
                "trading_date": str(today_ist),
                "latest_intraday_time_ist": str(today_frame.index[-1]),
                "latest_price": round(float(latest_close.iloc[-1]), 2),
                "volume": volume,
                "source": "5-minute intraday candles",
            }
        )

    if not current_day_rows:
        current_day_rows = current_day_rows_from_daily

    last_completed_trading_day = None
    if last_completed_dates:
        last_completed_trading_day = max(last_completed_dates)

    current_trading_day = None
    if intraday_dates:
        current_trading_day = max(intraday_dates)
    elif current_day_rows_from_daily:
        current_trading_day = str(today_ist)

    return {
        "generated_at_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "today_ist": str(today_ist),
        "last_completed_trading_day": last_completed_trading_day,
        "current_trading_day": current_trading_day,
        "current_day_note": (
            "Current day volume is live/intraday if market data is available. "
            "During market hours it can change until market close."
        ),
        "last_completed_day_top_5_by_volume": _format_volume_rows(last_day_rows),
        "current_day_top_5_by_volume": _format_volume_rows(current_day_rows),
    }


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_last_and_current_day_nifty_volumes",
            "description": (
                "Fetch top 5 Nifty 50 companies by traded volume for the last "
                "completed trading day and the current trading day so far."
            ),
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
                "You are a stock market data assistant. Use the tool result only. "
                "Do not invent dates, companies, prices, or volumes. "
                "Always mention the generated_at_ist value, the last completed "
                "trading day date, and the current trading day date. "
                "Explain clearly that current day volume may change while the "
                "market is open."
            ),
        },
        {
            "role": "user",
            "content": (
                "Give me the top 5 Nifty 50 companies by volume for the last "
                "completed trading day and current trading day. Include dates."
            ),
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
            if tool_call.function.name == "get_last_and_current_day_nifty_volumes":
                tool_result = get_last_and_current_day_nifty_volumes()

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": json.dumps(tool_result),
                    }
                )

        final_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
        )

        print(final_response.choices[0].message.content)
    else:
        print(assistant_message.content)


if __name__ == "__main__":
    run_agent()
