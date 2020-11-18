
from import_my_packages import *
from datetime import timedelta
import datetime

def get_daily_historical_stock_price_specific_statement(ticker, start_date, end_date, info_type):
  url_to_open = 'https://finance.yahoo.com/quote/' + ticker + '/'+ info_type + '?p=' + ticker
  print(url_to_open)
  read_data = ur.urlopen(url_to_open).read()
  soup= BeautifulSoup(read_data,'lxml')
  features = soup.find_all('div', class_='D(tbr)')

  headers = []
  temp_list = []
  label_list = []
  final = []
  index = 0
  #create headers
  for item in features[0].find_all('div', class_='D(ib)'):
      headers.append(item.text)
  #statement contents
  while index <= len(features)-1:
      #filter for each line of the statement
      temp = features[index].find_all('div', class_='D(tbc)')
      for line in temp:
          temp_list.append(line.text) #each item adding to a temporary list
      final.append(temp_list) #temp_list added to final list
      temp_list = [] #clear temp_list
      index+=1
  df = pd.DataFrame(final[1:])
  df.columns = headers


  def convert_to_numeric(column): #function to make all values numerical
      final_col = pd.to_numeric([i.replace('-','') for i in [i.replace(',','') for i in column]])
      return final_col

  for column in headers[1:]:
    df[column] = convert_to_numeric(df[column])
    final_df = df.fillna('-')
  return(final_df)



def get_daily_historical_stock_price(ticker, start_date, end_date):
  output = []
  statements = ['financials', 'balance-sheet', 'cash-flow']
  for statement_type in statements:
    x = get_daily_historical_stock_price_specific_statement(ticker, start_date, end_date, statement_type)
    output.append(x)
  return(output) # returns income statement df, balance sheet df, cash flow df



def get_historic_stock_prices(ticker, start_date, end_date):
  data = yf.download([ticker],'2015-1-1')
  df = pd.DataFrame(data.loc[start_date:end_date])
  df.columns = ["STOCK_PRICE_" + i.replace(" ", "_") for i in df.columns]
  return(df)



def get_FFM5(start_date, end_date):
  df = pd.read_csv("ffm5factorsdaily.csv")
  df['Date'] = df['Date'].astype(str)
  df['year'] = df['Date'].str[0:4]
  df['month'] = df['Date'].str[4:6]
  df['day'] = df['Date'].str[6:]
  df['datestring'] = df['year'] + df['month'] + df['day']
  df['Date_Column'] = pd.to_datetime(df['datestring'], infer_datetime_format=True)
  df = df[df['Date_Column'] >= start_date]
  df = df[df['Date_Column'] <= end_date]
  df.drop(['Date', 'year', 'month', 'day', 'datestring'], inplace = True, axis = 1)
  return(df)




def get_vix_index(start_date, end_date):
  end_date = end_date + timedelta(1)
  vix_data = yf.download("^VIX", start=start_date, end=end_date)
  df = pd.DataFrame(vix_data.loc[start_date:end_date])
  df.drop("Volume", inplace = True, axis = 1)
  df.columns = ["VIX_" + i.replace(" ", "_") for i in df.columns]
  return(df)



def get_composite_indices(start_date, end_date):
  NASDAQ_data = yf.download("^IXIC", start=start_date, end=end_date) # NASDAQ composite
  NASDAQ_df = pd.DataFrame(NASDAQ_data.loc[start_date:end_date])
  NASDAQ_df.drop("Volume", inplace = True, axis = 1)
  NASDAQ_df.columns = ["NASDAQ_" + i.replace(" ", "_") for i in NASDAQ_df.columns]

  DOW_data = yf.download("^DJA", start=start_date, end=end_date) # DOW composite
  DOW_df = pd.DataFrame(DOW_data.loc[start_date:end_date])
  DOW_df.drop("Volume", inplace = True, axis = 1)
  DOW_df.columns = ["DOW_" + i.replace(" ", "_") for i in DOW_df.columns]

  SP_data = yf.download("^GSPC", start=start_date, end=end_date) # S&P composite
  SP_df = pd.DataFrame(SP_data.loc[start_date:end_date])
  SP_df.drop("Volume", inplace = True, axis = 1)
  SP_df.columns = ["SP_" + i.replace(" ", "_") for i in SP_df.columns]

  df = pd.merge(NASDAQ_df,DOW_df, how='outer', left_index=True, right_index=True)
  df = pd.merge(df,SP_df, how='outer', left_index=True, right_index=True)

  return( df)




def create_numeric_training_data(ticker, start_date, end_date):
  ### Historic Stock Prices ###
  historic_stock_prices = get_historic_stock_prices(ticker, start_date, end_date)

  # ### VIX ###
  vix_df = get_vix_index(start_date, end_date)

  # ### Composite - Dow, Nasdaq, S&P ###
  composite_df = get_composite_indices(start_date, end_date)

  ### Fama French ###
  fama_french_df_big = get_FFM5(start_date, end_date)

  ### Merge ###
  merged_df = pd.merge(historic_stock_prices,vix_df, how='outer', left_index=True, right_index=True)
  merged_df = pd.merge(merged_df,composite_df, how='outer', left_index=True, right_index=True)
  merged_df['Date_Column'] = merged_df.index
  merged_df = pd.merge(merged_df, fama_french_df_big, how='outer',left_on = 'Date_Column', right_on = 'Date_Column')#, right_on = 'Date_Column')# left_index=True, right_index=True)
  merged_df = merged_df.set_index('Date_Column')

  for f in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
    merged_df[f] = merged_df[f].copy().shift(1)

  ### Edited after we decided to predict return ###
  merged_df['TARGET'] = (merged_df['STOCK_PRICE_Open'] - merged_df['STOCK_PRICE_Close']) / (merged_df['STOCK_PRICE_Open'])

  cols_to_keep = ['TARGET', 'STOCK_PRICE_Open','VIX_Open', 'NASDAQ_Open', 'DOW_Open', 'SP_Open', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
  merged_df = merged_df[cols_to_keep]

  merged_df.dropna(inplace = True, axis = 0)
  print(merged_df)

  return(merged_df)
