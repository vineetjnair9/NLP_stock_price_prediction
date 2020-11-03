from import_my_packages import * 
#only has tf-idf support right now, add more embeddings later
class Featurizer(object):
    """class to preprocess and tokenize large amounts of text
    and ultimately create textual features."""
    
    def __init__(self, lang = 'english'):
        """class instantiator."""
        self.stopset = set(stopwords.words(lang))
        self.corpus = set()
        self.tfidf = None
        
    def preprocess(self, input_str):
        """method to preprocess articles.
        removes punctuation, digits & stopwords
        for the given input string.
        
        Args:
            input_str :: str
                text to be preprocessed

        Returns:
            preprocessed string. Also stores 
            the cleaned string into corpus class variable
        """

        input_str = input_str.lower()

        #fastest way to remove/replace characters in python
        digits_table = str.maketrans('', '', string.digits)
        punct_table = str.maketrans('', '', string.punctuation)


        #can add more tables for future reference

        tables = [digits_table, punct_table]
        for t in tables:
            input_str = input_str.translate(t)

        #handling stopwords
        input_str = ' '.join([word for word in input_str.split() if word not in self.stopset])
        
        if input_str not in self.corpus:
            self.corpus.add(input_str)
        
        return input_str
    
    def tfidf_fit(self, use_idf = True, max_df = 1.0, min_df = 1, max_features = None):
        """generates the document term frequency
        matrix for the stored training corpus. Can serve as simple
        NLP baseline for the regression task.
        
        Args:
            use_idf :: bool
                whether or not to use idf weights. if
                set to False, simply uses tf weights
                
            max_df/min_df :: float or int
                if in [0,1], represents proportion 
                of terms to ignore with respect to corpus 

            max_features :: int
                considers only the "max_features" most frequent
                terms, if specified
                
        Returns:
            df :: DataFrame
                document term frequency matrix for the corpus
        """
        #params of the vectorizer
        tfidf = TfidfVectorizer(lowercase = False, 
                        preprocessor = self.preprocess,
                        stop_words = None,
                        ngram_range=(1,1),
                        tokenizer = None,
                        max_df = max_df, min_df = min_df,
                        max_features = max_features,
                        use_idf = use_idf)
        
        features = tfidf.fit_transform(self.corpus)
        self.tfidf = tfidf 
        df = pd.DataFrame(features.todense(), columns=tfidf.get_feature_names())

        return df
    
    def tfidf_transform(self, test_corpus):
        """transforms articles according to
        trained tfidf model. Use on out-of-sample articles to obtain
        textual tfidf features. 
        
        Args:
            test_corpus :: list(str)
                list of articles to transform into tfidf features
                
        Returns:
            X_test :: DataFrame
                articles transformed into features
        """
        
        if self.tfidf is None:
            return 'Must fit an tfidf model on training corpus first'
        
        X_test = self.tfidf.transform(test_corpus)
        X_test = pd.DataFrame(data=X_test.toarray(), columns= self.tfidf.get_feature_names())
        
        return X_test