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
A python script, backtest.py, was used to simulate trading for the past 2 year period using the dynamic strategy defined above and compared to buying and holding BTC for the same time period.  Past BTC prices were access via Coinbase API.  With an initial investment of $10,000 USD holding BTC for 2 years resulted in a final equity of $32,300 USD, a 223% gain.  With the same $10,000 USD invest the dynamic strategy resulted in a final equity of $47,400, a 374% gain.  Backtesting shows the hybrid strategy outperforms buy and holding BTC by 67% over the past two year period.
<img width="1887" height="929" alt="image" src="https://github.com/user-attachments/assets/bad13a2d-9b2f-4b2a-9f15-84e9769007c2" />

## BTC Trading Application
This trading application leverages the Alpaca API to support both paper and live trading environments, offering a seamless transition between simulation and real-world execution. The core strategy is dynamically defined through LLM-based prompting, enabling flexible, real-time adjustments. To ensure reliability, a rules-based fallback system is implemented for uninterrupted operation in the event of LLM access issues. The app is deployed on an AWS EC2 instance, ensuring continuous 24/7 availability and robust performance. The application is currently trading on an Alpaca paper account.

### Features
-   24/7 paper/live BTC trading on Alpaca with interactive dasboard
-   Dynamic strategy switching when bull or bear market is identified
-   LLM based action explanation
-   Toggle to switch between strict rules based strategy or LLM prompt based strategy
-   Realtime charting of account equity and BTCUSD price
-   Logging of all trades and errors
-   Daily and on-demand gmail report of P/L, open and closed trades
<img width="1889" height="899" alt="image" src="https://github.com/user-attachments/assets/a63388b2-4269-4725-91f4-12a51b9cf830" />

### Known issues
-  In LLM trading mode the application occassionally executes trades before a new buy/sell signal appears or ignores a buy/sell signal
  
## Next Steps
- Continue to running paper trading account to monitor applicaiton and strategy peformance
- Experiment with different LLM prompts and models to improve robustness of LLM trading mode to avoid false or ignored buy/sell signals
- Real-time email notification of new trades and application errors
- Add dashboard option to select gmail report frequency
- Add dashboard option to select charting time period and toggle to select between showing Equity, BTC price or both
