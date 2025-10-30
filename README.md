# Bitcoin-Trading-Agent

## Project Overview
A smart bitcoin trading system designed to operate with minimal human supervision and continuously adapt to changing market conditions.

## Goals
- Run 24/7 and deploy in a cloud environment.
- Adapt continuously to market conditions, ideally with the help of a lightweight LLM
- Read the text aloud with optional voice cloning
- Integrate feature into a single web based app.
   
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
- Voice Synthesis:  Neural voice cloning model via Chatterbox
- Web Framework: Flask
- Containerization: Docker
- Web hosting:  AWS EC2
### BTC Trading App Building and Deployment:
- IDE: VS
- Python Version: 3.10+
- Voice Synthesis:  Neural voice cloning model via Chatterbox
- Web Framework: Flask
- Containerization: Docker
- Web hosting:  AWS EC2

#### Running app Locally:
1. Start Flask App
```bash
python monreader.py
```

2. Access in Browser
```arduino
http://localhost:5000
```

#### Running with Docker
```bash
docker build -t monreader-app .
docker run -p 5000:5000 monreader-app
```

## Strategy
### Backtesting

## Application Features

## Next Steps

## Conclusions
