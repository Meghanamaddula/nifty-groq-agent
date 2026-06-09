import yfinance as yf
import pandas as pd


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


def get_top_5_by_volume():
    tickers = list(NIFTY_50.keys())

    data = yf.download(
        tickers=tickers,
        period="10d",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    rows = []

    for ticker in tickers:
        try:
            stock_data = data[ticker].dropna()
            stock_data = stock_data[stock_data["Volume"] > 0]

            if stock_data.empty:
                continue

            latest_row = stock_data.iloc[-1]
            latest_date = stock_data.index[-1].date()

            rows.append({
                "ticker": ticker,
                "company": NIFTY_50[ticker],
                "date": latest_date,
                "close": latest_row["Close"],
                "volume": int(latest_row["Volume"]),
            })

        except Exception as e:
            print(f"Could not fetch {ticker}: {e}")

    result = pd.DataFrame(rows)
    result = result.sort_values(by="volume", ascending=False).head(5)

    return result


if __name__ == "__main__":
    top_5 = get_top_5_by_volume()
    print(top_5.to_string(index=False))
