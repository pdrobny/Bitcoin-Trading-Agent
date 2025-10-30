# Bitcoin-Trading-Agent

## Project Overview
A smart bitcoin trading system designed to operate with minimal human supervision and continuously adapt to changing market conditions.

## Goals
- Run 24/7 and deploy in a cloud environment.
- Adapt continuously to market conditions, ideally with the help of a lightweight LLM
- Maintain a log of all trades
- Automatically send an email report via Gmail
   
## Project Structure
```BTC-Trading-App
├── app # folder containing files needed for app dockerization and deployment
│  ├── templates
│  │  ├── index.html # web page formatting template for flask app
│  ├── app.py # btc trading app code
│  ├── example.env # example of .env file for Alpaca, Gemini, Google login info and app configurations, must be updated with personal account login info prior to running app
│  ├── Dockerfile # containerization file
│  ├── requirements.txt # list of required installs
├── backtest.py # python script for strategy 2 year backtesting using Coinbase data
├── README.md
```
##  Installation and Setup
### Backtesting:
- IDE: VS
- Python Version: 3.10+
- Libraries:  time, numpy, pandas, ccxt, zoneinfo, plotly
### BTC Trading App Building and Deployment:
- IDE: VS
- Python Version: 3.10+
- Trading API: Alpaca
- Containerization: Docker
- Web hosting:  AWS EC2

#### Running app Locally:
1. Start Flask App
```bash
python app.py
```

2. Access in Browser
```arduino
http://localhost:5000
```

#### Running with Docker
```bash
docker build -t btc-bot:local .
docker run -d --name btc-bot ` 
--env-file .env `
-p 5000:5000 `
-v "${PWD}\data_local:/app/data" `
-restart unless-stopped `
btc-bot:local
```

## Strategy
The goal of this BTC trading app is to outperform buy and holding BTC.  The strategy is to maximize exposure to bull periods by using a long-term holding strategy, then switch to a short-term swing trading strategy to reduce exposure to bear periods and look for opportunities for small gains.  For this application a bull period is identified when the EMA(250) crosses above and remains above the EMA(500) on a 1-hour timeframe.  A bear period is defined when the EMA250 crosses below and remains below the EMA500 on a 1-hour timeframe.  The app will buy at the beginning of a bull period and sell at the end of a bull period.  During a bear period the app will buy and hold when the CCI(20) crosses above and remains above the CCI_MA(20) on a 1-hour timeframe and sell once the CCI(20) crosses below the CCI_MA(20).  All transactions will be for 100% of available equity with some buffer for fees and market movement.

BTC charting timeframe:  1 hour
App Glossary:
 EMA: Exponential Moving average.
 CCI:  The Commodity Channel Index is a momentum-based technical indicator that measures a security's price variation from its statistical mean. It helps traders identify overbought or oversold conditions, trend reversals, and divergences by comparing the current price to a historical average.
 CCI_MA:  The moving average of CCI.
 
### Backtesting

## Application Features

## Next Steps

## Conclusions
