from scraper_module import *

s = scraper(search_terms = ['GE stock news']) #init
s.make_df() #creates df, access with s.df
s.df
