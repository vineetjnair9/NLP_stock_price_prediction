
from import_my_packages import * 

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





### Fama French Factors ###
def get_Fama_French_Factors(ticker, start_date, end_date):

  # Define Helper Function 1 #
  def price2ret(prices,retType='simple'):
      if retType == 'simple':
          ret = (prices/prices.shift(1))-1
      else:
          ret = np.log(prices/prices.shift(1))
      return ret
  # Define Helper Function 2 #
  def assetPriceReg(df_stk):
      import pandas_datareader.data as web  # module for reading datasets directly from the web
      
      # Reading in factor data
      df_factors = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench')[0]
      
      df_factors.rename(columns={'Mkt-RF': 'MKT'}, inplace=True)
      df_factors['MKT'] = df_factors['MKT']/100 
      df_factors['SMB'] = df_factors['SMB']/100
      df_factors['HML'] = df_factors['HML']/100
      df_factors['RMW'] = df_factors['RMW']/100
      df_factors['CMA'] = df_factors['CMA']/100
      
      df_stock_factor = pd.merge(df_stk,df_factors,left_index=True,right_index=True) # Merging the stock and factor returns dataframes together
      df_stock_factor['XsRet'] = df_stock_factor['Returns'] - df_stock_factor['RF'] # Calculating excess returns

      # Running CAPM, FF3, and FF5 models.
      CAPM = smf.ols(formula = 'XsRet ~ MKT', data=df_stock_factor).fit(cov_type='HAC',cov_kwds={'maxlags':1})
      FF3 = smf.ols( formula = 'XsRet ~ MKT + SMB + HML', data=df_stock_factor).fit(cov_type='HAC',cov_kwds={'maxlags':1})
      FF5 = smf.ols( formula = 'XsRet ~ MKT + SMB + HML + RMW + CMA', data=df_stock_factor).fit(cov_type='HAC',cov_kwds={'maxlags':1})

      CAPMtstat = CAPM.tvalues
      FF3tstat = FF3.tvalues
      FF5tstat = FF5.tvalues

      CAPMcoeff = CAPM.params
      FF3coeff = FF3.params
      FF5coeff = FF5.params

      # DataFrame with coefficients and t-stats
      results_df = pd.DataFrame({'CAPMcoeff':CAPMcoeff,'CAPMtstat':CAPMtstat,
                                'FF3coeff':FF3coeff, 'FF3tstat':FF3tstat,
                                'FF5coeff':FF5coeff, 'FF5tstat':FF5tstat},
      index = ['Intercept', 'MKT', 'SMB', 'HML', 'RMW', 'CMA'])


      dfoutput = summary_col([CAPM,FF3, FF5],stars=True,float_format='%0.4f',
                    model_names=['CAPM','FF3','FF5'],
                    info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                              'Adjusted R2':lambda x: "{:.4f}".format(x.rsquared_adj)}, 
                              regressor_order = ['Intercept', 'MKT', 'SMB', 'HML', 'RMW', 'CMA'])

      return(pd.DataFrame(results_df))

  # Begin Current Function
  df = get_historic_stock_prices(ticker, start_date, end_date)
  df['Returns'] = price2ret(df[['STOCK_PRICE_Adj_Close']])
  df_regOutput = assetPriceReg(df)
  
  ### Test Run ###
  fama_french_df = df_regOutput
  capm_intercept = fama_french_df.at['Intercept', 'CAPMcoeff']
  capm_MKT = fama_french_df.at['MKT', 'CAPMcoeff']
  ff3_intercept = fama_french_df.at['Intercept', 'FF3coeff']
  ff3_mkt = fama_french_df.at['MKT', 'FF3coeff']
  ff3_smb = fama_french_df.at['SMB', 'FF3coeff']
  ff5_intercept = fama_french_df.at['Intercept', 'FF5coeff']
  ff5_mkt = fama_french_df.at['MKT', 'FF5coeff']
  ff5_smb = fama_french_df.at['SMB', 'FF5coeff']
  ff5_HML = fama_french_df.at['HML', 'FF5coeff']
  ff5_RMW = fama_french_df.at['RMW', 'FF5coeff']

  CAPM = [capm_intercept, capm_MKT]
  FF3 = [ff3_intercept, ff3_mkt, ff3_smb]
  FF5 = [ff5_intercept, ff5_mkt, ff5_smb, ff5_HML, ff5_RMW]
  return(CAPM, FF3, FF5)

def get_vix_index(start_date, end_date):
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


def get_historical_options_data(ticker, start_date, end_date):
  return(1)



def create_numeric_training_data(ticker, start_date, end_date):
  ### Historic Stock Prices ###
  historic_stock_prices = get_historic_stock_prices(ticker, start_date, end_date)

  ### VIX ###
  vix_df = get_vix_index(start_date, end_date)

  ### Composite - Dow, Nasdaq, S&P ###
  composite_df = get_composite_indices(start_date, end_date)

  ### Financial Statements ###
  income_statement_df, balance_sheet_df, cash_flow_statement_df = get_daily_historical_stock_price(ticker, start_date, end_date)

  ### Fama French ###
  fama_french_factors_CAPM, fama_french_factors_FF3, fama_french_factors_FF5 = get_Fama_French_Factors(ticker, start_date, end_date)

  ### Merge ###
  merged_df = pd.merge(historic_stock_prices,vix_df, how='outer', left_index=True, right_index=True)
  merged_df = pd.merge(merged_df,composite_df, how='outer', left_index=True, right_index=True)
  return(merged_df)


