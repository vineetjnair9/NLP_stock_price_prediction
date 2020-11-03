from scraper_module import *
from get_financial_data import * 
from import_my_packages import * 
from featurizer_module import *
##################### PLOT SETTINGS #####################
font_dict = {'size' : 40, 'family': 'serif'}
font_dict_legend = {'size' : 20, 'family': 'serif'}
tick_size = 30
###################################################################

home_directory = os.getcwd()


ticker = 'AMZN'
start_date = datetime(2019, 10, 30)
start_date_string = "10/30/2019" #2019/10/30"
end_date = datetime(2019, 11,5)
end_date_string = "11/05/2019" #2019/11/5"


get_numeric_data = 0
get_articles_data = 0

if get_numeric_data == 1:
	training_numeric_df = create_numeric_training_data(ticker, start_date, end_date)

if get_articles_data == 1:
	s = scraper(search_terms = ['GE'], date_from = start_date_string, date_to = end_date_string) #init
	s.make_df() #creates df, access with s.df
	print(s.df)
	s.df.to_csv(home_directory + "/Articles_K_Means/articles_for_kmeans.csv")


# I gotta use scrape period. 
# s.scrape_period




nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
print(stopwords.words('english'))





training_corpus = ['this is information about document one', 
                   '$3NoW, so!me^ information about the >,/?seco0nd! 488492document',
                  'this is the last article news!!']
f = Featurizer()
for a in training_corpus:
    f.preprocess(a)

print('pre-processed articles/text:', f.corpus)
print('')
doc_matrix = f.tfidf_fit(use_idf = True)
doc_matrix
#new testing examples to be used as features
test_corpus = ['one test instance', 
               'this is the second one, hopefully contains some information',
              'Final $#document742 with [information]!}']

f.tfidf_transform(test_corpus)

#example of pre-processing

stopset = set(stopwords.words('english'))
def preprocess(input_str):
    """removes punctuation and digits
    for the given input string."""
    
    input_str = input_str.lower()
    
    #fastest way to remove/replace characters in python
    digits_table = str.maketrans('', '', string.digits)
    punct_table = str.maketrans('', '', string.punctuation)
    
    
    #can add more tables for future reference

    tables = [digits_table, punct_table]
    for t in tables:
        input_str = input_str.translate(t)

    #handling stopwords
    input_str = ' '.join([word for word in input_str.split() if word not in stopset])
    
    return input_str

print(preprocess('This&/ $!!is a #42 {[important]} +70TEST|'))