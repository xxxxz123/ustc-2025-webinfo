import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import collections

class TextTokenizer:
    def __init__(self, enable_phrase_detection=False, min_phrase_freq=2):
        try:
            # 检查NLTK数据是否已下载
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("正在下载NLTK数据...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        self.enable_phrase_detection = enable_phrase_detection
        self.min_phrase_freq = min_phrase_freq
        self.common_phrases = self._load_common_phrases()
    
    def _load_common_phrases(self):
        """加载基础常用词组"""
        base_phrases = [
            "machine learning", "data science", "artificial intelligence",
            "deep learning", "neural networks", "web development",
            "computer science", "meetup group", "big data",
            "cloud computing", "cyber security", "user experience",
            "computer security", "online privacy", "technology exploration",
            "monthly meetings", "study group", "hands on",
            "real world", "open source", "software development"
        ]
        return set(base_phrases)
    
    def extract_phrases_statistical(self, texts, max_phrases=30):
        """基于统计方法从文本中提取常用词组"""
        if not texts:
            return set()
            
        all_bigrams = []
        all_trigrams = []
        
        for text in texts:
            # 清理文本
            cleaned_text = self.clean_text(text)
            tokens = word_tokenize(cleaned_text)
            tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
            
            # 提取二元组和三元组
            bigrams = list(nltk.bigrams(tokens))
            trigrams = list(nltk.trigrams(tokens))
            
            all_bigrams.extend(bigrams)
            all_trigrams.extend(trigrams)
        
        # 统计频率
        bigram_freq = collections.Counter(all_bigrams)
        trigram_freq = collections.Counter(all_trigrams)
        
        # 选择高频词组
        phrases = set()
        
        # 添加高频二元组
        for bigram, count in bigram_freq.most_common(max_phrases//2):
            if count >= self.min_phrase_freq:
                phrase = f"{bigram[0]}_{bigram[1]}"
                phrases.add(phrase)
        
        # 添加高频三元组
        for trigram, count in trigram_freq.most_common(max_phrases//2):
            if count >= self.min_phrase_freq:
                phrase = f"{trigram[0]}_{trigram[1]}_{trigram[2]}"
                phrases.add(phrase)
        
        return phrases
    
    def extract_phrases_pos(self, text):
        """基于词性标注提取名词短语"""
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
        
        sentences = nltk.sent_tokenize(text)
        phrases = set()
        
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(tokens)
            
            # 提取名词短语模式 (形容词+名词, 名词+名词等)
            grammar = r"""
                NP: {<JJ.*>*<NN.*>+}  # 形容词* + 名词+
                NP: {<NN.*>+<NN.*>+}   # 名词+名词
            """
            try:
                chunker = nltk.RegexpParser(grammar)
                tree = chunker.parse(pos_tags)
                
                for subtree in tree.subtrees():
                    if subtree.label() == 'NP':
                        phrase_tokens = [token for token, pos in subtree.leaves()]
                        if len(phrase_tokens) >= 2:
                            phrase = '_'.join(phrase_tokens)
                            phrases.add(phrase)
            except Exception:
                # 如果解析失败，跳过该句子
                continue
        
        return phrases
    
    def adaptive_phrase_detection(self, texts):
        """自适应词组检测 - 结合多种方法"""
        all_phrases = set(self.common_phrases)
        
        # 方法1: 统计方法
        statistical_phrases = self.extract_phrases_statistical(texts)
        all_phrases.update(statistical_phrases)
        
        # 方法2: 词性标注方法
        for text in texts:
            pos_phrases = self.extract_phrases_pos(text)
            all_phrases.update(pos_phrases)
        
        return all_phrases
    
    def expand_contractions(self, text):
        """扩展缩写形式，如you're -> you are"""
        contractions = {
            "you're": "you are",
            "i'm": "i am",
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "we're": "we are",
            "they're": "they are",
            "that's": "that is",
            "what's": "what is",
            "who's": "who is",
            "where's": "where is",
            "when's": "when is",
            "why's": "why is",
            "how's": "how is",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "won't": "will not",
            "wouldn't": "would not",
            "can't": "cannot",
            "couldn't": "could not",
            "shouldn't": "should not",
            "mightn't": "might not",
            "mustn't": "must not",
            "i've": "i have",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            "i'll": "i will",
            "you'll": "you will",
            "he'll": "he will",
            "she'll": "she will",
            "it'll": "it will",
            "we'll": "we will",
            "they'll": "they will",
            "i'd": "i would",
            "you'd": "you would",
            "he'd": "he would",
            "she'd": "she would",
            "it'd": "it would",
            "we'd": "we would",
            "they'd": "they would"
        }
        
        for contraction, expansion in contractions.items():
            text = re.sub(r'\b' + contraction + r'\b', expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def clean_text(self, text):
        """清理文本，去除特殊字符、数字等"""
        # 扩展缩写
        text = self.expand_contractions(text)
        
        # 转换为小写
        text = text.lower()
        
        # 移除URL
        text = re.sub(r'http\S+', '', text)
        
        # 移除电子邮件
        text = re.sub(r'\S+@\S+', '', text)
        
        # 移除数字
        text = re.sub(r'\d+', '', text)
        
        # 移除标点符号和特殊字符，但保留单词间的空格
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 移除多余的空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_with_phrases(self, texts):
        """使用词组检测进行分词（批量处理）"""
        if not self.enable_phrase_detection:
            # 如果不启用词组检测，使用普通分词
            return [self.tokenize(text) for text in texts], set()
        
        # 首先从所有文本中学习常用词组
        learned_phrases = self.adaptive_phrase_detection(texts)
        print(f"学习到 {len(learned_phrases)} 个常用词组")
        
        results = []
        for text in texts:
            cleaned_text = self.clean_text(text)
            tokens = []
            
            # 替换学习到的词组
            temp_text = cleaned_text
            for phrase in sorted(learned_phrases, key=len, reverse=True):
                phrase_space = phrase.replace('_', ' ')
                if phrase_space in temp_text:
                    temp_text = temp_text.replace(phrase_space, phrase)
            
            # 分词剩余部分
            remaining_tokens = word_tokenize(temp_text)
            for token in remaining_tokens:
                if '_' in token:
                    # 这是检测到的词组
                    tokens.append(token)
                else:
                    # 普通词，过滤停用词
                    if token not in self.stop_words and len(token) > 1:
                        tokens.append(token)
            
            results.append(tokens)
        
        return results, learned_phrases
    
    def tokenize_single_with_phrases(self, text, learned_phrases=None):
        """对单个文本使用词组检测进行分词"""
        if not self.enable_phrase_detection or not learned_phrases:
            # 如果不启用词组检测或没有学习到的词组，使用普通分词
            return self.tokenize(text)
        
        cleaned_text = self.clean_text(text)
        tokens = []
        
        # 替换学习到的词组
        temp_text = cleaned_text
        for phrase in sorted(learned_phrases, key=len, reverse=True):
            phrase_space = phrase.replace('_', ' ')
            if phrase_space in temp_text:
                temp_text = temp_text.replace(phrase_space, phrase)
        
        # 分词剩余部分
        remaining_tokens = word_tokenize(temp_text)
        for token in remaining_tokens:
            if '_' in token:
                # 这是检测到的词组
                tokens.append(token)
            else:
                # 普通词，过滤停用词
                if token not in self.stop_words and len(token) > 1:
                    tokens.append(token)
        
        return tokens
    
    def tokenize(self, text):
        """基础分词处理（不包含词组检测）"""
        cleaned_text = self.clean_text(text)
        tokens = word_tokenize(cleaned_text)
        # 过滤停用词和短词
        filtered_tokens = [token for token in tokens if token not in self.stop_words and len(token) > 1]
        return filtered_tokens