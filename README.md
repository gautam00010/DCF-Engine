# 🏦 DCF-Engine: Autonomous Equity Research Pipeline

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![Finance](https://img.shields.io/badge/domain-Equity%20Research-purple)

## 📌 Executive Summary
**DCF-Engine** is an automated, institutional-grade valuation pipeline. It acts as an autonomous equity research analyst, dynamically fetching financial statements, projecting future cash flows, and calculating the intrinsic value of top equities using a Discounted Cash Flow (DCF) model.

The system is managed by a state-saving Python scheduler and runs via GitHub Actions CI/CD, guaranteeing fresh weekly scenario analysis (Base, Bull, Bear) outputted directly to Excel.

## ⚙️ System Architecture
1. **The Scheduler (`scheduler.py`):** Acts as the brain. Reads a dynamic `universe.csv` of tickers, selects the stalest asset, and orchestrates the valuation to ensure systematic coverage without redundant API calls.
2. **Data Ingestion (`valuation/data_fetcher.py`):** Connects to the Financial Modeling Prep (FMP) API to extract point-in-time Income Statements, Balance Sheets, Cash Flow Statements, and WACC metrics.
3. **Valuation Engine (`valuation/dcf_model.py`):** The mathematical core. Projects 5-year Free Cash Flow (FCF), applies a Gordon Growth Terminal Value, and discounts cash flows to present value. 
4. **Report Generator (`valuation/report_generator.py`):** Uses `pandas` and `openpyxl` to format the mathematical outputs into a color-coded, multi-sheet Excel workbook detailing the valuation bridge and implied upside/downside.

## 📊 Output
For every ticker analyzed, the engine dynamically generates:
* `reports/{Ticker}/{Ticker}_DCF_Report_{Date}.xlsx`
* Scenario breakdowns highlighting Enterprise Value to Equity Value bridges.
* Implied Margin of Safety vs. Current Market Price.

## 🚀 How to Run Locally
1. Clone the repository and install dependencies:
   `pip install -r requirements.txt`
2. Set your API key:
   `export FMP_API_KEY="your_api_key_here"`
3. Run the autonomous scheduler:
   `python scheduler.py`

*Note: All core logic is heavily tested. Run `pytest` to execute the offline test suite.*
