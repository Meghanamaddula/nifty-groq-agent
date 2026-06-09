import os
import json
import yfinance as yf
from groq import Groq
from dotenv import load_dotenv


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


COMPANY_TO_TICKER = {
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "hdfc bank": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "sbi": "SBIN.NS",
    "bharti airtel": "BHARTIARTL.NS",
    "itc": "ITC.NS",
    "larsen": "LT.NS",
    "lt": "LT.NS",
    "axis bank": "AXISBANK.NS",
    "kotak bank": "KOTAKBANK.NS",
    "wipro": "WIPRO.NS",
    "tata motors": "TMPV.NS",
    "tata steel": "TATASTEEL.NS",
    "sun pharma": "SUNPHARMA.NS",
    "maruti": "MARUTI.NS",
}


def find_ticker(company_name):
    name = company_name.lower().strip()

    for key, ticker in COMPANY_TO_TICKER.items():
        if key in name or name in key:
            return ticker

    if not name.endswith(".ns"):
        return name.upper() + ".NS"

    return name.upper()


def analyze_company(company_name):
    ticker_symbol = find_ticker(company_name)
    ticker = yf.Ticker(ticker_symbol)

    info = ticker.info
    history = ticker.history(period="5d")
    news = ticker.news[:5] if ticker.news else []

    current_price = None
    if not history.empty:
        current_price = round(float(history.iloc[-1]["Close"]), 2)

    pe_ratio = info.get("trailingPE")
    market_cap = info.get("marketCap")
    company_long_name = info.get("longName", company_name)

    news_items = []

    for item in news:
        news_items.append({
            "title": item.get("title"),
            "publisher": item.get("publisher"),
            "link": item.get("link"),
        })

    return {
        "input_company": company_name,
        "detected_ticker": ticker_symbol,
        "company_name": company_long_name,
        "current_price": current_price,
        "pe_ratio": pe_ratio,
        "market_cap": market_cap,
        "recent_news": news_items,
    }


tools = [
    {
        "type": "function",
        "function": {
            "name": "analyze_company",
            "description": "Fetch PE ratio, market cap, current price, and recent news for an Indian stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "Name of the company to analyze, for example Reliance or Infosys.",
                    }
                },
                "required": ["company_name"],
            },
        },
    }
]


def run_company_agent(company_name):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an educational stock analysis assistant. "
                "Use the tool result. Do not invent data. "
                "Explain PE ratio, market cap, current price, and news sentiment. "
                "Give a cautious investment view: positive, neutral, or negative. "
                "Always say this is not financial advice."
            ),
        },
        {
            "role": "user",
            "content": f"Analyze {company_name}. Tell me PE, market cap, current price, news summary, and whether it looks good to invest right now.",
        },
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    assistant_message = response.choices[0].message
    messages.append(assistant_message)

    if assistant_message.tool_calls:
        for tool_call in assistant_message.tool_calls:
            if tool_call.function.name == "analyze_company":
                args = json.loads(tool_call.function.arguments)
                tool_result = analyze_company(args["company_name"])

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
    company = input("Enter company name: ")
    run_company_agent(company)