import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

class TextNormalizer:
    def __init__(self, use_lemmatization=True):
        self.use_lemmatization = use_lemmatization
        self._download_nltk_data()
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def _download_nltk_data(self):
        """下载必要的NLTK数据"""
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("正在下载NLTK stopwords数据...")
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("正在下载NLTK wordnet数据...")
            nltk.download('wordnet', quiet=True)
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            print("正在下载NLTK POS tagger数据...")
            nltk.download('averaged_perceptron_tagger', quiet=True)
    
    def remove_stopwords(self, tokens):
        """去除停用词"""
        return [token for token in tokens if token not in self.stop_words and len(token) > 1]
    
    def stem_tokens(self, tokens):
        """词干提取"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens):
        """词形还原"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def normalize(self, tokens):
        """对分词后的token进行规范化处理"""
        # 去除停用词
        filtered_tokens = self.remove_stopwords(tokens)
        
        # 词形还原或词干提取
        if self.use_lemmatization:
            normalized_tokens = self.lemmatize_tokens(filtered_tokens)
        else:
            normalized_tokens = self.stem_tokens(filtered_tokens)
        
        return normalized_tokens