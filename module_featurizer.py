from import_my_packages import *
#only has tf-idf support right now, add more embeddings later

##### in case of missing packages, import the below #####

# from string import digits
# #for building document term matrices
# from sklearn.feature_extraction.text import TfidfVectorizer

# #nltk stopwords
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from gensim.models import Word2Vec

# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#only has tf-idf support right now, add more embeddings later
#Look into NMF factorization

class Featurizer(object):
    """class to preprocess corpuses or text
    and ultimately create textual features."""

    def __init__(self, lang = 'english'):
        """class instantiator."""
        self.stopset = set(stopwords.words(lang))
        self.corpus = set()
        self.tfidf = None
        #for dimensionality reduction
        self.scaler = None
        self.pca = None
        #word2vec
        self.w2v = None

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

    def PCA_fit(self, X_train, n_components=0.9):
        """fits PCA on training sample. stores the PCA
        object and appropriate scaler to use on the test set.

        Args:
            X_train :: DataFrame or np.array
                Training set, can be relatively large matrix
                of chosen unigram features.

            n_components :: float
                number of components needed to explain
                n_components% of the variation in X_train

        Returns:
            PCs :: np.array
                principal components of the training set
        """

        sc = StandardScaler()
        X_train_std = sc.fit_transform(X_train)
        self.scaler = sc

        #obtaining the PCs from training data
        pca_fit = PCA(n_components)
        PCs = pca_fit.fit_transform(X_train_std)
        self.pca = pca_fit

        return PCs

    def PCA_transform(self, X_test):
        """transform out-of-sample examples
        with PCs from training sample."""

        if (self.scaler == None) or (self.pca == None):
            return "must fit PCA to the training sample first"

        X_test_std = sc.transform(X_test)

        return self.pca.transform(X_test_std)

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
        X_test = pd.DataFrame(data=X_test.toarray(), columns= f.tfidf.get_feature_names())

        return X_test

    def word2vec(self, pre_train_path = None, size = 100, min_count = 1, window = 5):
        """loads or train Word2Vec embeddings
        using class corpus.

        Args:
            pre_train_path :: str
                path to file containing pre-trained embeddings,
                e.g. GoogleNews embeddings

            size :: int
                dimension of the word embedding,
                e.g. # of features

            min_count :: int
                ignores words occuring less than min_count

        Stores:
            self.w2v :: Word2Vec object
        """

        #using pre_trained embeddings
        if pre_train_path:
            pre_trained = KeyedVectors.load_word2vec_format(pre_train_path, binary=True)

            self.w2v = pre_trained
            return 'stored pre-trained word embeddings'

        #tokenizing articles
        tokenized_articles = []
        for a in f.corpus:
            #maybe use another tokenizer, not sure if gensim.utils is best
            tokenized_articles.append(gensim.utils.simple_preprocess(a))

        model = Word2Vec(sentences = tokenized_articles, sg = 0, size = size,
                         min_count = min_count, workers = -1, window = window)
        self.w2v = model

        return 'trained word embeddings on corpus'

    def article2vec(self, article):
        """given an aticle/text,
        returns a word embedding for that particular text
        using the self.w2v model (obtained with word2vec).

        Args:
            article :: str
                article text

        Returns:
            vec :: np.array
                word embedding of the text
        """

        if self.w2v is None:
            return "must load/train a word2vec model first"

        vec = np.zeros(shape = (self.w2v.vector_size))

        txt = self.preprocess(article)
        #convert into tokens, can perhaps be more picky here
        tokens = gensim.utils.simple_preprocess(txt)

        for t in tokens:
            #check if token in vocab
            if t in self.w2v.wv:
                #maybe consider using a weighted word2vec embedding
                vec += self.w2v.wv[t]

        return vec
