from GoogleNews import GoogleNews
from newspaper import Config
from newspaper import Article
import pandas as pd
import datetime as dt
# import matplotlib.pyplot as plt
import nltk
# nltk.download('punkt')

import requests
import multiprocessing
#beautiful soup is a noteworthy API to try

#SQL
# import pyodbc
from sqlalchemy import event, create_engine
from collections import defaultdict


#just leave this as is for now
dd = defaultdict(lambda: 1)

def date_util(date):
    """converts datetime object to 
    string and vice versa
    
    Args:
        date :: str or datetime object
            if str, must be in MM/DD/YYYY format
    """
    
    if (type(date) == str):
        return dt.datetime.strptime(date,"%m/%d/%Y")
    
    if (type(date) != str):
        return date.strftime("%m/%d/%Y")
        
class scraper(object):
    """scrapes relevant google articles, given a list of search terms.
       Uses GoogleNews, will extend support for pyGoogleNews 
    """
    
    today = dt.datetime.now().strftime("%m/%d/%Y")
    
    def __init__(self, date_from = today, date_to = today, search_terms = []):
        """ class instantiator.
        
        Args:
            date_from :: str
                date string in format MM/DD/YYYY, will only parse 
                articles released on that date until date_to
                
            date_to :: str
                date string in same format as above. articles
                dated after this date will not be parsed.
            
            search_terms :: list(str)
                list of search terms to parse on google. relevancy of each
                article will be assessed via default dictionary. 
        """
        
        self.date_from = date_from
        self.date_to = date_to
        self.search_terms = search_terms
        #to be stored in methods
        self.search_info = None
        self.data = None
        self.df = None
    
        #dbs and date scraped, for the write_sql method
        self.dbs = []
        self.dates_scraped = set()
        
    def set_date(self, date):
        """Utility to function to change
            the date class variable. Useful for scraping.
            
        Args:
            date :: str
                date string in format MM/DD/YYYY     
        """
        
        self.date_from, self.date_to = date, date
        
        return None
    
    def get_links(self, pages = 1):
        """obtains all relevant links from the search,
            for each company.
            
        Args:
            pages :: int
                number of google pages to search resuts from
                
        Stores:
            links :: dict(list[dict])
                dictionaries of list, keys being search terms
                and values being relevant information (e.g. URL)
        """
        
        gnews = GoogleNews(start=self.date_from, end=self.date_to)
        links = {}
        
        #obtaining all the URLs
        for s in self.search_terms:
            gnews.search(s)
            for p in range(1,pages+1):
                gnews.getpage(p)
                result = gnews.result() #stores values until cleared
            
            links[s] = result
            gnews.clear()
            
        #removing irrelevant links
        for s in self.search_terms:
            tmp = []
            num = dd[s] #number of relevant terms in search_terms
            rel_str = ' '.join(s.lower().split()[:num]) #relevant string

            for d in links[s]:
                #selection criterion, e.g. if search term  
                #is 'apple news', then want to subset based on 'apple' rather than 'apple news'
                #--> filter with first word of each search term
                if rel_str in d['desc'].lower(): 
                    tmp.append(d)   
            links[s] = tmp 
        
        self.search_info = links
        
        return None    
    
    def process_link(self, link = None, nlp = False):
        """processes the linksobtain by get_links() method, extracts
            both the text and a summary of the article with newspaper package
            
        Args:
            link :: str
                URL of links stored in the dictionary returned by get_links()
            nlp :: bool
                Whether or not to perform nlp on the text of the link. This extracts
                a summary of the text, but is a somewhat expensive operation.
        
        Returns:
            article :: 'article' object
                object that contains parsed properties for the link, such as
                summary, text and date.
        """
        
        #parameters for the processing
        config = Config()
        config.fetch_images = False #no need for images
        config.memoize_articles = False #no need for article caching
        
        try:
            article = Article(link, language = "en", config = config)
            article.download()
            article.parse()
            if nlp:  
                article.nlp() #extract summary as per the newspaper API
        except:
            return None
    
        return article  
    
    def store_data(self, search_info = None, nlp = False):
        """ stores data for all links, for each in search term.
            e.g. date, the summary, text...
           
            Args:
                links :: dict(list[dicts])
                    dictionary that containts URLs for each of our
                    search terms, e.g. returned by get_links() method.
                    
                nlp :: bool
                    Whether a summary was extracted in the process_links()
                    method.
                   
            Stores:
                res :: dict(list[dicts])
                    dictionary that stores info for all our searches. Can be used
                    to make DataFrame easily, and then upload to SQL database later.
                    Info stored for each link: date|search_term|link|summary|text
        """
        
        if search_info is None:
            search_info = self.search_info
        
        res = {} #will build df using a dictionary
        
        for s in self.search_terms: #iterate over search terms
            res[s] = []
            
            #relevant string
            num = dd[s] 
            rel_str = ' '.join(s.lower().split()[:num]) 
            
            for info in search_info[s]: #iterate over links
                tmp = {}
                #only need one date assuming we run this class daily
                tmp['date'] = self.date_to 
                tmp['core_search_term'] = rel_str #to handle keys appropriately
                tmp['link'] = info['link'] 
                
                #process the link, use try clause in case failure to process
                a = self.process_link(tmp['link'])
                try:
                    tmp['text'] = a.text #might need to narrow depending on length of text
                    if nlp:
                        try:
                            tmp['summary'] = a.summary
                        except:
                            tmp['summary'] = None
                except:
                    tmp['text'] = None
        
                #store result
                res[s].append(tmp)
        
        self.data = res
        print('search data stored, {}'.format(self.date_to))
        
        return None
    
    def make_df(self, res = None):
        """ returns dataframe containing all relevant
            results for the day, for all our searches.
           
            Args:
                res :: dict(list[dicts])
                    dictionary stored after calling self.store_data()
                   
            
            Stores:
                df :: DataFrame
                    DataFrame of results for the given searches
        """
        
        self.get_links()
        self.store_data()
        
        if res is None:
            res = self.data
        
        df = pd.DataFrame()
        for s in self.search_terms:
            tmp = pd.DataFrame(res[s])
            df = df.append(tmp)
         
        df = df.reset_index()
        df.drop(columns= ['index'], inplace=True)
        self.df = df
        
        return None
    
    def write_sql(self, db_name):
        #change this, remove exists ect...
        """creates SQL table in internship server.
           Will only work once make_df() has been called.
        
            Args:
                db_name :: str
                    name of the database
        """
        
        if self.df is None:
            print('no data: call make_df() first')
            return None
    
        else:
            server = None
            sql_engine = create_engine(server)
            
            if db_name in self.dbs:
                self.df.to_sql(db_name, sql_engine, if_exists = 'append')
                print('appended to database {}'.format())
               
            else:
                self.df.to_sql(db_name, sql_engine)
                print('created new database {}'.format(db_name))
                self.dbs.append(db_name)
            
        return None
    
    def scrape_period(db_name, begin, end):
        """scrape news for entire period, from begin to
        (including) end, for all search terms provided. Handles
        dates previously scraped for a given scraper instance.
        
        Args:
            db_name :: str
                name of the database to create/append to
            
            begin, end :: str
                date string in format MM/DD/YYYY    
        """
        
        #change this to create single df rather than append to
        #SQL server
        while date_util(begin) <= date_util(end):
            if begin not in self.dates_scraped:
        
                #set date to scrape
                self.set_date(begin)
                
                #obtain data
                self.make_df()
                self.write_sql(db_name)
                
                #add to scraped dates
                self.dates_scraped.add(begin)
            #increment date by one day
            begin = date_util(date_util(begin) + dt.timedelta(days=1))
        
        return None   