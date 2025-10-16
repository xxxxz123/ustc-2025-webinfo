import os
import math
from collections import defaultdict, namedtuple

# 定义文档信息结构
DocumentInfo = namedtuple('DocumentInfo', ['doc_id', 'doc_name'])

class SkipListNode:
    """跳表节点类"""
    def __init__(self, doc_id):
        self.doc_id = doc_id  # 文档ID
        self.occurrences = 0  # 词项在文档中出现的频率
        self.next = None      # 下一个节点
        self.skip = None      # 跳表指针
        self.skip_distance = 0  # 跳表指针跨越的距离

class InvertedIndex:
    """倒排表实现类"""
    def __init__(self):
        # 倒排表结构: {词项: 跳表头节点}
        self.index = defaultdict(SkipListNode)
        # 文档信息: {文档ID: DocumentInfo对象}
        self.documents = {}
        # 词项文档频率: {词项: 包含该词项的文档数}
        self.document_frequencies = defaultdict(int)
        # 文档长度: {文档ID: 文档中的词项总数}
        self.document_lengths = defaultdict(int)
        # 跳表层级因子
        self.skip_list_factor = 4  # 每4个节点创建一个跳表指针

    def build_from_file(self, file_path):
        """从文件构建倒排表"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                # 跳过注释行
                if line.startswith('#'):
                    continue
                
                # 解析行内容
                try:
                    parts = line.strip().split('|', 2)
                    if len(parts) != 3:
                        print(f"警告: 第{line_num+1}行格式不正确，跳过")
                        continue
                    
                    doc_id = parts[0].strip()
                    doc_name = parts[1].strip()
                    normalized_text = parts[2].strip()
                    
                    # 保存文档信息
                    self.documents[doc_id] = DocumentInfo(doc_id=doc_id, doc_name=doc_name)
                    
                    # 分词
                    terms = normalized_text.split()
                    self.document_lengths[doc_id] = len(terms)
                    
                    # 统计词频
                    term_freq = defaultdict(int)
                    for term in terms:
                        term_freq[term] += 1
                    
                    # 更新倒排表
                    for term, freq in term_freq.items():
                        self.add_posting(term, doc_id, freq)
                        
                except Exception as e:
                    print(f"处理第{line_num+1}行时出错: {e}")
        
        # 构建跳表指针
        self._build_skip_pointers()
        
        print(f"倒排表构建完成。")
        print(f"包含 {len(self.documents)} 个文档。")
        print(f"包含 {len(self.index)} 个不同的词项。")

    def add_posting(self, term, doc_id, frequency=1):
        """向倒排表添加一个文档-词项对"""
        # 如果是新词项，初始化跳表头节点
        if term not in self.index:
            # 创建一个虚拟头节点
            self.index[term] = SkipListNode(doc_id="HEAD")
            self.index[term].next = None
        
        # 找到插入位置
        current = self.index[term]
        while current.next is not None and current.next.doc_id < doc_id:
            current = current.next
        
        # 检查是否已经存在该文档ID
        if current.next is not None and current.next.doc_id == doc_id:
            # 如果已存在，更新频率
            current.next.occurrences = frequency
            return
        
        # 创建新节点并插入
        new_node = SkipListNode(doc_id)
        new_node.occurrences = frequency
        new_node.next = current.next
        current.next = new_node
        
        # 更新文档频率
        self.document_frequencies[term] += 1

    def _build_skip_pointers(self):
        """为倒排表构建跳表指针"""
        for term, head in self.index.items():
            current = head.next  # 跳过虚拟头节点
            count = 0
            skip_node = head
            skip_count = 0
            
            while current is not None:
                count += 1
                skip_count += 1
                
                # 每skip_list_factor个节点创建一个跳表指针
                if count % self.skip_list_factor == 0:
                    skip_node.skip = current
                    skip_node.skip_distance = skip_count
                    skip_node = current
                    skip_count = 0
                
                current = current.next
            
            # 确保最后一个节点的跳表指针为空
            if skip_node:
                skip_node.skip = None

    def remove_document(self, doc_id):
        """删除倒排表中的指定文档"""
        # 检查文档是否存在
        if doc_id not in self.documents:
            print(f"文档 {doc_id} 不存在")
            return False

        # 从所有词项的倒排列表中删除该文档
        for term in list(self.index.keys()):
            head = self.index[term]
            current = head
            found = False
            
            # 查找并删除该文档节点
            while current.next is not None:
                if current.next.doc_id == doc_id:
                    # 删除节点
                    current.next = current.next.next
                    found = True
                    break
                current = current.next
            
            # 如果删除了节点，更新文档频率
            if found:
                self.document_frequencies[term] -= 1
                # 如果词项不再出现在任何文档中，删除该词项
                if self.document_frequencies[term] == 0:
                    del self.index[term]
                    del self.document_frequencies[term]
        
        # 删除文档信息
        del self.documents[doc_id]
        del self.document_lengths[doc_id]
        
        # 重新构建跳表指针
        self._build_skip_pointers()
        
        print(f"文档 {doc_id} 已成功删除")
        return True
    
    def remove_term(self, term):
        """删除倒排表中的指定词项"""
        # 检查词项是否存在
        if term not in self.index:
            print(f"词项 '{term}' 不存在")
            return False
        
        # 删除词项及其相关信息
        del self.index[term]
        if term in self.document_frequencies:
            del self.document_frequencies[term]
        
        print(f"词项 '{term}' 已成功删除")
        return True
    
    def search_term(self, term):
        """搜索单个词项，返回包含该词项的文档列表"""
        if term not in self.index:
            return []
        
        results = []
        current = self.index[term].next  # 跳过虚拟头节点
        
        while current is not None:
            doc_info = self.documents.get(current.doc_id, None)
            if doc_info:
                results.append({
                    'doc_id': current.doc_id,
                    'doc_name': doc_info.doc_name,
                    'frequency': current.occurrences
                })
            current = current.next
        
        return results

    def search_with_skip(self, term):
        """使用跳表指针搜索单个词项，返回包含该词项的文档列表"""
        if term not in self.index:
            return []
        
        results = []
        current = self.index[term]  # 从虚拟头节点开始
        
        # 使用跳表指针加速搜索
        while current is not None:
            if current.doc_id != "HEAD":  # 跳过虚拟头节点
                doc_info = self.documents.get(current.doc_id, None)
                if doc_info:
                    results.append({
                        'doc_id': current.doc_id,
                        'doc_name': doc_info.doc_name,
                        'frequency': current.occurrences
                    })
            
            # 优先使用跳表指针
            if current.skip is not None:
                current = current.skip
            else:
                # 没有跳表指针时，使用普通指针
                current = current.next if current.doc_id == "HEAD" else current.next
        
        return results

    def boolean_search(self, query):
        """布尔搜索，支持AND、OR、NOT操作符"""
        # 简化实现，只支持AND操作
        terms = query.lower().split()
        if not terms:
            return []
        
        # 获取第一个词项的结果
        results = set([item['doc_id'] for item in self.search_term(terms[0])])
        
        # 对剩余词项进行AND操作
        for term in terms[1:]:
            current_results = set([item['doc_id'] for item in self.search_term(term)])
            results.intersection_update(current_results)
        
        # 返回文档信息
        return [{
            'doc_id': doc_id,
            'doc_name': self.documents[doc_id].doc_name
        } for doc_id in results]

    def tf_idf_search(self, query, top_k=5):
        """TF-IDF搜索，返回相关性最高的top_k个文档"""
        terms = query.lower().split()
        if not terms:
            return []
        
        # 计算每个文档的TF-IDF分数
        scores = defaultdict(float)
        total_docs = len(self.documents)
        
        for term in terms:
            if term not in self.index:
                continue
            
            # 计算IDF
            df = self.document_frequencies[term]
            idf = math.log(total_docs / (df + 1)) if df > 0 else 0
            
            # 获取包含该词项的文档
            postings = self.search_term(term)
            
            # 计算TF-IDF并累加分数
            for posting in postings:
                doc_id = posting['doc_id']
                tf = posting['frequency'] / self.document_lengths[doc_id]  # 归一化的TF
                scores[doc_id] += tf * idf
        
        # 按分数排序并返回top_k个结果
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [{
            'doc_id': doc_id,
            'doc_name': self.documents[doc_id].doc_name,
            'score': score
        } for doc_id, score in sorted_results]

    def save_index(self, file_path):
        """保存倒排表到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            # 先保存文档信息
            f.write("# DOCUMENTS\n")
            for doc_id, doc_info in self.documents.items():
                f.write(f"{doc_id}|{doc_info.doc_name}|{self.document_lengths[doc_id]}\n")
            
            # 保存倒排表
            f.write("# INVERTED_INDEX\n")
            for term, head in sorted(self.index.items()):
                postings = []
                current = head.next
                while current is not None:
                    postings.append(f"{current.doc_id}:{current.occurrences}")
                    current = current.next
                if postings:
                    f.write(f"{term}|{','.join(postings)}\n")

    def load_index(self, file_path):
        """从文件加载倒排表"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        self.index.clear()
        self.documents.clear()
        self.document_frequencies.clear()
        self.document_lengths.clear()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            section = None
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('# DOCUMENTS'):
                    section = 'documents'
                    continue
                elif line.startswith('# INVERTED_INDEX'):
                    section = 'index'
                    continue
                
                if section == 'documents':
                    parts = line.split('|', 2)
                    if len(parts) == 3:
                        doc_id, doc_name, doc_length = parts
                        self.documents[doc_id] = DocumentInfo(doc_id=doc_id, doc_name=doc_name)
                        self.document_lengths[doc_id] = int(doc_length)
                elif section == 'index':
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        term, postings_str = parts
                        postings = postings_str.split(',')
                        
                        # 添加所有文档ID到倒排表
                        for posting in postings:
                            doc_id, freq = posting.split(':', 1)
                            self.add_posting(term, doc_id, int(freq))
        
        # 重建跳表指针
        self._build_skip_pointers()

# 使用示例，倒排表结构
# index = {
#     'atheist': SkipListNode("HEAD") -> SkipListNode("12763", occurrences=2) -> SkipListNode("12929", occurrences=1) -> SkipListNode("13032", occurrences=1),
#     'meetup_group': SkipListNode("HEAD") -> SkipListNode("12763", occurrences=1) -> SkipListNode("21458", occurrences=1) -> SkipListNode("570", occurrences=1),
#     # 更多词项...
# }

# documents = {
#     '12763': DocumentInfo(doc_id='12763', doc_name='The Boston Atheists Meetup Group'),
#     '12929': DocumentInfo(doc_id='12929', doc_name='Rhode Island Atheist Society'),
#     # 更多文档...
# }

# document_frequencies = {
#     'atheist': 3,
#     'meetup_group': 3,
#     # 更多词项...
# }

# document_lengths = {
#     '12763': 15,  # 该文档包含15个词项
#     '12929': 30,  # 该文档包含30个词项
#     # 更多文档...
# }
if __name__ == "__main__":
    # 创建倒排表实例
    index = InvertedIndex()
    
    # 从文件构建倒排表
    file_path = '../generated_documents/sample_event_retrieval.txt'
    try:
        index.build_from_file(file_path)
        
        # 测试搜索
        print("\n测试搜索'theist':")
        results = index.search_term('atheist')
        for result in results:
            print(f"  - {result['doc_id']}: {result['doc_name']} (频率: {result['frequency']})")
        
        print("\n测试布尔搜索'atheist group':")
        results = index.boolean_search('atheist group')
        for result in results:
            print(f"  - {result['doc_id']}: {result['doc_name']}")
        
        print("\n测试TF-IDF搜索'boston meetup':")
        results = index.tf_idf_search('boston meetup', top_k=3)
        for result in results:
            print(f"  - {result['doc_id']}: {result['doc_name']} (分数: {result['score']:.4f})")
        
        # 保存倒排表
        index.save_index('../generated_documents/inverted_index.txt')
        print("\n倒排表已保存到文件。")

        # 测试删除
        print("\n测试删除文档功能:")
        index.remove_document('12763')
        
        print("\n再次搜索'theist':")
        results = index.search_term('atheist')
        for result in results:
            print(f"  - {result['doc_id']}: {result['doc_name']} (频率: {result['frequency']})")
        
        print("\n测试删除词项功能:")
        index.remove_term('group')
        
        print("\n搜索已删除的词项'group':")
        results = index.search_term('group')
        if not results:
            print("  - 未找到任何结果")

        # 保存倒排表
        index.save_index('../generated_documents/inverted_index.txt')
        print("\n倒排表已保存到文件。")
        
    except Exception as e:
        print(f"发生错误: {e}")