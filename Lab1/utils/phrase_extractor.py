import spacy
import yake
from rake_nltk import Rake

class PhraseExtractor:
    def __init__(self):
        self.nlp = None
        self.initialize_tools()
    
    def initialize_tools(self):
        """初始化外部工具"""
        try:
            # 尝试加载spacy模型
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("请先安装spacy英文模型: python -m spacy download en_core_web_sm")
                self.nlp = None
        except ImportError:
            print("某些工具未安装，将使用基础方法")
    
    def extract_with_spacy(self, text):
        """使用spacy提取名词短语"""
        if not self.nlp:
            return set()
        
        doc = self.nlp(text)
        phrases = set()
        
        # 提取名词短语
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:
                phrase = chunk.text.lower().replace(' ', '_')
                phrases.add(phrase)
        
        # 提取实体
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'EVENT', 'WORK_OF_ART']:
                if len(ent.text.split()) >= 2:
                    phrase = ent.text.lower().replace(' ', '_')
                    phrases.add(phrase)
        
        return phrases
    
    def extract_with_yake(self, text, max_phrases=10):
        """使用YAKE算法提取关键词短语"""
        try:
            kw_extractor = yake.KeywordExtractor(
                lan="en", 
                n=3,  # 最大短语长度
                dedupLim=0.9,
                top=max_phrases
            )
            keywords = kw_extractor.extract_keywords(text)
            phrases = set()
            for kw, score in keywords:
                if len(kw.split()) >= 2:  # 只保留多词短语
                    phrase = kw.lower().replace(' ', '_')
                    phrases.add(phrase)
            return phrases
        except ImportError:
            return set()
    
    def extract_with_rake(self, text, max_phrases=10):
        """使用RAKE算法提取关键词短语"""
        try:
            r = Rake()
            r.extract_keywords_from_text(text)
            phrases = set()
            for phrase in r.get_ranked_phrases()[:max_phrases]:
                if len(phrase.split()) >= 2:
                    normalized_phrase = phrase.lower().replace(' ', '_')
                    phrases.add(normalized_phrase)
            return phrases
        except ImportError:
            return set()
    
    def ensemble_extraction(self, text, methods=None):
        """集成多种方法提取短语"""
        if methods is None:
            methods = ['spacy', 'yake', 'rake']
        
        all_phrases = set()
        
        if 'spacy' in methods:
            all_phrases.update(self.extract_with_spacy(text))
        
        if 'yake' in methods:
            all_phrases.update(self.extract_with_yake(text))
        
        if 'rake' in methods:
            all_phrases.update(self.extract_with_rake(text))
        
        return all_phrases