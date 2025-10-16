import math
import os
import struct
from collections import defaultdict, namedtuple

# 定义文档信息结构
DocumentInfo = namedtuple('DocumentInfo', ['doc_id', 'doc_name'])

class SkipListNode:
    """优化的跳表节点类，包含词项位置信息"""
    def __init__(self, doc_id):
        self.doc_id = doc_id  # 文档ID
        self.occurrences = 0  # 词项在文档中出现的频率
        self.position_list = []  # 词项在文档中出现的位置列表
        self.next = None      # 下一个节点
        self.skip = None      # 跳表指针
        self.skip_distance = 0  # 跳表指针跨越的距离

class OptimizedInvertedIndex:
    """优化的倒排表实现类"""
    def __init__(self, compression_method="delta"):
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
        # 压缩方法选择: "delta", "variable_byte", "none"
        self.compression_method = compression_method
        # 用于存储压缩后的数据
        self.compressed_data = {}
        # 记录压缩前后的大小以进行比较
        self.original_size = 0
        self.compressed_size = 0

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
                    
                    # 分词并记录位置
                    terms = normalized_text.split()
                    self.document_lengths[doc_id] = len(terms)
                    
                    # 统计词频和位置
                    term_positions = defaultdict(list)
                    for pos, term in enumerate(terms):
                        term_positions[term].append(pos)
                    
                    # 更新倒排表
                    for term, positions in term_positions.items():
                        self.add_posting(term, doc_id, len(positions), positions)
                        
                except Exception as e:
                    print(f"处理第{line_num+1}行时出错: {e}")
        
        # 构建跳表指针
        self._build_skip_pointers()
        
        # 压缩索引
        if self.compression_method != "none":
            self.compress_index()
        
        print(f"优化倒排表构建完成。")
        print(f"包含 {len(self.documents)} 个文档。")
        print(f"包含 {len(self.index)} 个不同的词项。")
        
        if self.compression_method != "none":
            self.report_compression_ratio()

    def add_posting(self, term, doc_id, frequency=1, positions=None):
        """向倒排表添加一个文档-词项对，并包含位置信息"""
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
            # 如果已存在，更新频率和位置
            current.next.occurrences = frequency
            current.next.position_list = positions or []
            return
        
        # 创建新节点并插入
        new_node = SkipListNode(doc_id)
        new_node.occurrences = frequency
        new_node.position_list = positions or []
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

    def phrase_search(self, phrase):
        """短语搜索实现"""
        terms = phrase.lower().split()
        if not terms:
            return []
        
        # 获取第一个词的所有文档
        first_term_docs = self.search_term(terms[0])
        if not first_term_docs:
            return []
        
        # 对每个文档检查短语匹配
        results = []
        
        for doc_result in first_term_docs:
            doc_id = doc_result['doc_id']
            # 获取所有词在该文档中的位置
            term_positions = {}
            all_terms_present = True
            
            for term in terms:
                positions = self._get_term_positions_in_doc(term, doc_id)
                if not positions:
                    all_terms_present = False
                    break
                term_positions[term] = positions
            
            if not all_terms_present:
                continue
            
            # 检查是否存在连续位置的短语
            if self._check_phrase_positions(terms, term_positions):
                results.append({
                    'doc_id': doc_id,
                    'doc_name': self.documents[doc_id].doc_name
                })
        
        return results

    def _get_term_positions_in_doc(self, term, doc_id):
        """获取词项在指定文档中的位置列表"""
        if term not in self.index:
            return []
        
        current = self.index[term].next
        while current is not None:
            if current.doc_id == doc_id:
                # 如果数据已压缩，解压位置信息
                if term in self.compressed_data and doc_id in self.compressed_data[term]:
                    return self._decompress_positions(term, doc_id)
                return current.position_list
            current = current.next
        
        return []

    def _check_phrase_positions(self, terms, term_positions):
        """检查是否存在连续位置的短语"""
        # 从第一个词的每个位置开始检查
        for pos in term_positions[terms[0]]:
            match = True
            # 检查后续每个词是否在正确的相对位置
            for i, term in enumerate(terms[1:], 1):
                if (pos + i) not in term_positions[term]:
                    match = False
                    break
            
            if match:
                return True
        
        return False

    def compress_index(self):
        """压缩倒排表索引"""
        self.compressed_data = {}
        self.original_size = 0
        self.compressed_size = 0
        
        for term, head in self.index.items():
            current = head.next
            term_data = {}
            
            while current is not None:
                doc_id = current.doc_id
                positions = current.position_list
                
                # 计算原始大小（近似值）
                self.original_size += len(str(doc_id)) + 4 + 4 * len(positions)
                
                if self.compression_method == "delta":
                    # 差值编码
                    compressed_positions = self._delta_encode(positions)
                    term_data[doc_id] = compressed_positions
                    self.compressed_size += len(str(doc_id)) + 4 + 4 * len(compressed_positions)
                    
                elif self.compression_method == "variable_byte":
                    # 可变字节编码
                    compressed_positions = self._variable_byte_encode(positions)
                    term_data[doc_id] = compressed_positions
                    self.compressed_size += len(str(doc_id)) + len(compressed_positions)
                
                current = current.next
            
            if term_data:
                self.compressed_data[term] = term_data

    def _delta_encode(self, positions):
        """差值编码实现"""
        if not positions:
            return []
            
        # 排序位置列表
        sorted_positions = sorted(positions)
        delta_encoded = [sorted_positions[0]]  # 第一个位置保持原值
        
        # 后续位置存储与前一个位置的差值
        for i in range(1, len(sorted_positions)):
            delta_encoded.append(sorted_positions[i] - sorted_positions[i-1])
            
        return delta_encoded

    def _delta_decode(self, delta_encoded):
        """差值解码实现"""
        if not delta_encoded:
            return []
            
        positions = [delta_encoded[0]]
        
        # 累积差值得到原始位置
        for i in range(1, len(delta_encoded)):
            positions.append(positions[i-1] + delta_encoded[i])
            
        return positions

    def _variable_byte_encode(self, numbers):
        """可变字节编码实现"""
        result = b''
        
        for num in numbers:
            bytes_list = []
            while True:
                bytes_list.insert(0, num % 128)
                if num < 128:
                    break
                num = num // 128
            bytes_list[-1] |= 128  # 设置结束标志位
            result += bytes(bytes_list)
            
        return result

    def _variable_byte_decode(self, encoded_bytes):
        """可变字节解码实现"""
        numbers = []
        current_num = 0
        
        for byte in encoded_bytes:
            if byte < 128:
                current_num = current_num * 128 + byte
            else:
                current_num = current_num * 128 + (byte - 128)
                numbers.append(current_num)
                current_num = 0
                
        return numbers

    def _decompress_positions(self, term, doc_id):
        """解压位置信息"""
        if term not in self.compressed_data or doc_id not in self.compressed_data[term]:
            return []
            
        compressed = self.compressed_data[term][doc_id]
        
        if self.compression_method == "delta":
            return self._delta_decode(compressed)
        elif self.compression_method == "variable_byte":
            return self._variable_byte_decode(compressed)
            
        return []

    def report_compression_ratio(self):
        """报告压缩率"""
        if self.original_size == 0:
            print("无法计算压缩率：原始数据大小为0")
            return
            
        compression_ratio = (1 - self.compressed_size / self.original_size) * 100
        print(f"压缩方法: {self.compression_method}")
        print(f"原始大小: 约 {self.original_size} 字节")
        print(f"压缩后大小: 约 {self.compressed_size} 字节")
        print(f"压缩率: {compression_ratio:.2f}%")

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
                    # 存储位置信息
                    positions_str = ":" + ",".join(map(str, current.position_list)) if current.position_list else ""
                    postings.append(f"{current.doc_id}:{current.occurrences}{positions_str}")
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
        self.compressed_data.clear()
        
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
                            # 解析位置信息
                            if ':' in posting:
                                parts = posting.split(':', 2)
                                doc_id = parts[0]
                                freq = int(parts[1])
                                positions = list(map(int, parts[2].split(','))) if len(parts) > 2 and parts[2] else []
                                self.add_posting(term, doc_id, freq, positions)
        
        # 重建跳表指针
        self._build_skip_pointers()
        
        # 压缩索引
        if self.compression_method != "none":
            self.compress_index()

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
                    # 同时从压缩数据中删除
                    if term in self.compressed_data:
                        del self.compressed_data[term]
            # 否则从压缩数据中删除该文档
            elif term in self.compressed_data and doc_id in self.compressed_data[term]:
                del self.compressed_data[term][doc_id]
        
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
        if term in self.compressed_data:
            del self.compressed_data[term]
        
        print(f"词项 '{term}' 已成功删除")
        return True

# 测试代码
if __name__ == "__main__":
    # 测试差值编码压缩
    print("\n=== 测试差值编码压缩 ===")
    delta_index = OptimizedInvertedIndex(compression_method="delta")
    delta_index.build_from_file('../generated_documents/sample_event_retrieval.txt')
    
    # 测试可变字节编码压缩
    print("\n=== 测试可变字节编码压缩 ===")
    vb_index = OptimizedInvertedIndex(compression_method="variable_byte")
    vb_index.build_from_file('../generated_documents/sample_event_retrieval.txt')
    
    # 测试短语搜索
    print("\n=== 测试短语搜索 ===")
    phrase = "boston meetup"
    print(f"搜索短语: '{phrase}'")
    results = delta_index.phrase_search(phrase)
    if results:
        for result in results:
            print(f"  - {result['doc_id']}: {result['doc_name']}")
    else:
        print("  未找到匹配结果")
        
    # 保存优化后的倒排表
    delta_index.save_index('../generated_documents/optimized_inverted_index.txt')
    print("\n优化后的倒排表已保存")