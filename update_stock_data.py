import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import glob
import pytz

def get_company_name_mapping():
    """Get mapping of stock symbols to company names from EQUITY.csv"""
    company_mapping = {}
    if os.path.exists('EQUITY.csv'):
        try:
            equity_df = pd.read_csv('EQUITY.csv')
            company_mapping = dict(zip(equity_df['SYMBOL'], equity_df['NAME OF COMPANY']))
        except Exception as e:
            print(f"Error reading EQUITY.csv: {str(e)}")
    return company_mapping

def get_all_stock_symbols():
    """Get all stock symbols from EQUITY.csv and existing data files"""
    symbols = set()
    
    # Get symbols from EQUITY.csv
    if os.path.exists('EQUITY.csv'):
        try:
            equity_df = pd.read_csv('EQUITY.csv')
            symbols.update(equity_df['SYMBOL'].tolist())
        except Exception as e:
            print(f"Error reading EQUITY.csv: {str(e)}")
    
    # Get symbols from existing data files
    csv_files = glob.glob('data/*_NS_data.csv')
    for file in csv_files:
        symbol = os.path.basename(file).replace('_NS_data.csv', '')
        symbols.add(symbol)
    
    return sorted(list(symbols))

def normalize_date(date_str):
    """Convert any date string to a timezone-naive datetime object in UTC"""
    try:
        # Parse the date string to datetime
        dt = pd.to_datetime(date_str)
        
        # If timezone aware, convert to UTC and make naive
        if dt.tzinfo is not None:
            dt = dt.tz_convert('UTC').tz_localize(None)
        return dt
    except:
        return None

def update_stock_data():
    # Get all stock symbols and company name mapping
    stock_symbols = get_all_stock_symbols()
    company_mapping = get_company_name_mapping()
    
    print("\nChecking and updating stock data files...")
    print("=" * 50)
    
    for symbol in stock_symbols:
        try:
            file_path = f'data/{symbol}_NS_data.csv'
            company_name = company_mapping.get(symbol, '')  # Get company name or empty string if not found
            
            # Initialize last_date to None
            last_date = None
            
            # If file exists, read last date
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['Date'] = df['Date'].apply(normalize_date)
                df = df.dropna(subset=['Date'])
                if not df.empty:
                    last_date = df['Date'].max()
            
            # Get current date (timezone naive in UTC)
            current_date = pd.Timestamp.now(pytz.UTC).normalize().tz_localize(None)
            
            print(f"\nProcessing {symbol}.NS")
            if last_date:
                print(f"Previous last date: {last_date.strftime('%Y-%m-%d')}")
            else:
                print("No existing data file")
            
            # Download data
            try:
                stock = yf.Ticker(f"{symbol}.NS")
                
                if last_date:
                    # Get data since last date
                    new_data = stock.history(start=last_date + pd.Timedelta(days=1))
                else:
                    # Get all available data
                    new_data = stock.history(period="max")
                
                if not new_data.empty:
                    # Prepare new data
                    new_data = new_data.reset_index()
                    new_data.columns = new_data.columns.str.capitalize()
                    
                    # Make dates timezone naive
                    new_data['Date'] = new_data['Date'].dt.tz_localize(None)
                    
                    # Add stock name and company name columns
                    new_data['Stocks_Name'] = symbol
                    new_data['Company_Name'] = company_name  # Add company name to each row
                    
                    if last_date:
                        # Append new data to existing file
                        df = pd.concat([df, new_data], ignore_index=True)
                        df = df.drop_duplicates(subset=['Date'], keep='last')
                        # Ensure company name is set for all rows
                        df['Company_Name'] = company_name
                    else:
                        # Create new file with all data
                        df = new_data
                    
                    # Sort by date
                    df = df.sort_values('Date')
                    
                    # Save data
                    os.makedirs('data', exist_ok=True)
                    df.to_csv(file_path, index=False)
                    
                    # Verify the update
                    updated_df = pd.read_csv(file_path)
                    updated_df['Date'] = updated_df['Date'].apply(normalize_date)
                    latest_date = updated_df['Date'].max()
                    
                    print(f"Updated with {len(new_data)} new entries")
                    print(f"Current last date: {latest_date.strftime('%Y-%m-%d')}")
                else:
                    print("No data available")
            except Exception as e:
                print(f"Error downloading data: {str(e)}")
                
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Update process completed.")

if __name__ == "__main__":
    update_stock_data() 