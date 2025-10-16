import time
import math
from optimized_inverted_index import OptimizedInvertedIndex
from collections import defaultdict

class OptimizedRetrievalTests:
    """使用优化倒排表进行检索任务的测试类"""
    def __init__(self, retrieval_file='../generated_documents/sample_event_retrieval.txt'):
        """初始化检索测试系统"""
        # 初始化不同压缩方法的倒排表
        self.delta_index = OptimizedInvertedIndex(compression_method="delta")
        self.vb_index = OptimizedInvertedIndex(compression_method="variable_byte")
        self.uncompressed_index = OptimizedInvertedIndex(compression_method="none")
        
        # 构建索引
        print("正在构建倒排表索引...")
        
        start_time = time.time()
        self.delta_index.build_from_file(retrieval_file)
        delta_time = time.time() - start_time
        
        start_time = time.time()
        self.vb_index.build_from_file(retrieval_file)
        vb_time = time.time() - start_time
        
        start_time = time.time()
        self.uncompressed_index.build_from_file(retrieval_file)
        uncompressed_time = time.time() - start_time
        
        print(f"索引构建时间:")
        print(f"  差值编码: {delta_time:.4f}秒")
        print(f"  可变字节编码: {vb_time:.4f}秒")
        print(f"  未压缩: {uncompressed_time:.4f}秒")
        
    # ---------------------- 布尔检索任务 ----------------------
    def boolean_search(self, index, query):
        """执行布尔检索"""
        # 简单解析布尔表达式
        terms = query.lower().split()
        operations = []
        current_terms = []
        
        # 提取操作符和词项
        i = 0
        while i < len(terms):
            if terms[i].upper() in ['AND', 'OR', 'NOT']:
                if current_terms:
                    operations.append(('TERM', ' '.join(current_terms)))
                    current_terms = []
                operations.append(('OP', terms[i].upper()))
            else:
                current_terms.append(terms[i])
            i += 1
        
        # 添加最后一组词项
        if current_terms:
            operations.append(('TERM', ' '.join(current_terms)))
        
        # 处理布尔查询
        results = set()
        current_op = None
        
        for op_type, value in operations:
            if op_type == 'TERM':
                term_docs = set()
                term_results = index.search_term(value)
                for result in term_results:
                    term_docs.add(result['doc_id'])
                
                if not results:
                    # 第一个词项
                    results = term_docs
                else:
                    # 根据操作符合并结果
                    if current_op == 'AND':
                        results = results.intersection(term_docs)
                    elif current_op == 'OR':
                        results = results.union(term_docs)
                    elif current_op == 'NOT':
                        results = results.difference(term_docs)
            elif op_type == 'OP':
                current_op = value
        
        return results
    
    def task_a1_complex_queries(self):
        """任务A.1: 设计复杂查询条件并分析不同处理顺序的影响"""
        print("\n=== 任务A.1: 复杂查询条件处理顺序分析 ===")
        
        # 设计3种复杂查询条件
        complex_queries = [
            "(boston OR rhode) AND atheist AND meetup",
            "computer_security OR electronic_gadgetry AND (internet_censorship OR online_privacy)",
            "atheist society AND (monthly_meetings NOT providence)"
        ]
        
        for query_idx, complex_query in enumerate(complex_queries, 1):
            print(f"\n查询 {query_idx}: {complex_query}")
            
            # 方法1: 先处理括号内的操作，再处理其他操作
            start_time = time.time()
            # 简化实现：手动模拟不同处理顺序
            if query_idx == 1:
                # 顺序1: 先处理OR
                or_results = self.boolean_search(self.delta_index, "boston OR rhode")
                # 模拟将OR结果与其他词项AND
                combined_results1 = set()
                for doc_id in or_results:
                    if self._doc_contains_terms(self.delta_index, doc_id, ['atheist', 'meetup']):
                        combined_results1.add(doc_id)
                order1_time = time.time() - start_time
                
                # 顺序2: 先处理AND
                start_time = time.time()
                and_results = self.boolean_search(self.delta_index, "atheist AND meetup")
                # 模拟将AND结果与其他词项OR
                combined_results2 = set()
                for doc_id in and_results:
                    if self._doc_contains_terms(self.delta_index, doc_id, ['boston', 'rhode']):
                        combined_results2.add(doc_id)
                order2_time = time.time() - start_time
                
            elif query_idx == 2:
                # 顺序1: 先处理括号内的OR和右侧AND
                start_time = time.time()
                inner_or = self.boolean_search(self.delta_index, "internet_censorship OR online_privacy")
                right_and = set()
                for doc_id in inner_or:
                    if self._doc_contains_terms(self.delta_index, doc_id, ['electronic_gadgetry']):
                        right_and.add(doc_id)
                combined_results1 = set()
                for doc_id in right_and:
                    combined_results1.add(doc_id)
                for doc_id in self.boolean_search(self.delta_index, "computer_security"):
                    combined_results1.add(doc_id)
                order1_time = time.time() - start_time
                
                # 顺序2: 先处理左侧的OR
                start_time = time.time()
                left_or = self.boolean_search(self.delta_index, "computer_security OR electronic_gadgetry")
                combined_results2 = set()
                for doc_id in left_or:
                    if self._doc_contains_terms(self.delta_index, doc_id, ['internet_censorship', 'online_privacy'], any_of=True):
                        combined_results2.add(doc_id)
                order2_time = time.time() - start_time
            else:
                # 顺序1: 先处理括号内的NOT
                start_time = time.time()
                not_providence = set()
                monthly_docs = set()
                for result in self.delta_index.search_term("monthly_meetings"):
                    monthly_docs.add(result['doc_id'])
                for result in self.delta_index.search_term("providence"):
                    monthly_docs.discard(result['doc_id'])
                not_providence = monthly_docs
                # 再处理AND
                combined_results1 = set()
                for doc_id in not_providence:
                    if self._doc_contains_terms(self.delta_index, doc_id, ['atheist', 'society']):
                        combined_results1.add(doc_id)
                order1_time = time.time() - start_time
                
                # 顺序2: 先处理左侧的AND
                start_time = time.time()
                atheist_society = self.boolean_search(self.delta_index, "atheist society")
                # 再处理与括号内的条件
                combined_results2 = set()
                for doc_id in atheist_society:
                    if doc_id in monthly_docs:
                        combined_results2.add(doc_id)
                order2_time = time.time() - start_time
            
            print(f"顺序1: {len(combined_results1)}个结果, 耗时: {order1_time*1000:.2f}毫秒")
            print(f"顺序2: {len(combined_results2)}个结果, 耗时: {order2_time*1000:.2f}毫秒")
            
            # 计算性能差异
            diff = abs(order1_time - order2_time)
            percentage = (diff / max(order1_time, order2_time)) * 100 if max(order1_time, order2_time) > 0 else 0
            print(f"性能差异: {percentage:.2f}%")
            
            # 分析结果
            if order1_time < order2_time:
                print("结论: 这种处理顺序更优，先处理选择性较大的操作可以减少后续操作的候选集")
            else:
                print("结论: 这种处理顺序更优，先处理选择性较小的操作可以更快地缩小搜索空间")
    
    def _doc_contains_terms(self, index, doc_id, terms, any_of=False):
        """检查文档是否包含所有指定词项"""
        found_count = 0
        for term in terms:
            found = False
            term_results = index.search_term(term)
            for result in term_results:
                if result['doc_id'] == doc_id:
                    found = True
                    found_count += 1
                    if any_of and found_count > 0:
                        return True
                    break
        return found_count == len(terms) or (any_of and found_count > 0)
    
    def task_a2_compression_efficiency(self):
        """任务A.2: 比较索引压缩前后的检索效率"""
        print("\n=== 任务A.2: 索引压缩前后检索效率对比 ===")
        
        # 测试查询
        queries = [
            "boston meetup",
            "atheist society",
            "computer_security internet_censorship"
        ]
        
        for query in queries:
            print(f"\n查询: {query}")
            
            # 未压缩索引检索
            start_time = time.time()
            uncompressed_results = self.boolean_search(self.uncompressed_index, query)
            uncompressed_time = time.time() - start_time
            
            # 差值编码索引检索
            start_time = time.time()
            delta_results = self.boolean_search(self.delta_index, query)
            delta_time = time.time() - start_time
            
            # 可变字节编码索引检索
            start_time = time.time()
            vb_results = self.boolean_search(self.vb_index, query)
            vb_time = time.time() - start_time
            
            print(f"  未压缩: {len(uncompressed_results)}个结果, 耗时: {uncompressed_time*1000:.2f}毫秒")
            print(f"  差值编码: {len(delta_results)}个结果, 耗时: {delta_time*1000:.2f}毫秒")
            print(f"  可变字节编码: {len(vb_results)}个结果, 耗时: {vb_time*1000:.2f}毫秒")
            
            # 计算性能提升百分比
            delta_improvement = ((uncompressed_time - delta_time) / uncompressed_time) * 100 if uncompressed_time > 0 else 0
            vb_improvement = ((uncompressed_time - vb_time) / uncompressed_time) * 100 if uncompressed_time > 0 else 0
            
            print(f"  差值编码相比未压缩性能{'提升' if delta_improvement > 0 else '下降'}: {abs(delta_improvement):.2f}%")
            print(f"  可变字节编码相比未压缩性能{'提升' if vb_improvement > 0 else '下降'}: {abs(vb_improvement):.2f}%")
            
        # 输出压缩率信息
        print("\n各索引压缩率信息:")
        print("差值编码索引:")
        self.delta_index.report_compression_ratio()
        print("\n可变字节编码索引:")
        self.vb_index.report_compression_ratio()
    
    def task_a3_phrase_search(self):
        """任务A.3: 短语检索分析"""
        print("\n=== 任务A.3: 短语检索分析 ===")
        
        # 设计短语检索
        phrases = [
            "boston meetup",
            "computer security",
            "monthly meetings",
            "atheist society"
        ]
        
        for phrase in phrases:
            print(f"\n搜索短语: '{phrase}'")
            
            start_time = time.time()
            results = self.delta_index.phrase_search(phrase)
            phrase_time = time.time() - start_time
            
            print(f"  短语检索结果数: {len(results)}")
            print(f"  耗时: {phrase_time*1000:.2f}毫秒")
            
            # 对比普通布尔检索
            start_time = time.time()
            boolean_results = self.boolean_search(self.delta_index, phrase)
            boolean_time = time.time() - start_time
            
            print(f"  布尔检索结果数: {len(boolean_results)}")
            print(f"  耗时: {boolean_time*1000:.2f}毫秒")
            
            # 计算精度提升
            if len(boolean_results) > 0:
                # 计算召回率
                recall = 1.0
                for res in results:
                    if res['doc_id'] not in boolean_results:
                        recall = len([r for r in results if r['doc_id'] in boolean_results]) / len(results)
                        break
                
                # 计算精确率
                precision = len(results) / len(boolean_results) if len(results) <= len(boolean_results) else 1.0
                print(f"  精确率: {precision*100:.2f}%")
                print(f"  召回率: {recall*100:.2f}%")
                print(f"  短语检索可以过滤掉仅包含词项但不按正确顺序出现的文档")
            
            # 显示部分结果
            if results:
                print("  部分结果:")
                for res in results[:3]:
                    print(f"    - {res['doc_id']}: {res['doc_name']}")
    
    def task_a4_skip_list_analysis(self):
        """任务A.4: 跳表指针步长分析"""
        print("\n=== 任务A.4: 跳表指针步长对存储和检索的影响 ===")
        
        # 测试步长
        step_sizes = [2, 4, 8, 16, 32]
        query = "boston meetup"
        
        # 保存原始步长
        original_step = self.delta_index.skip_list_factor
        
        # 存储不同步长的性能数据
        performance_data = []
        
        for step in step_sizes:
            # 设置新步长
            self.delta_index.skip_list_factor = step
            # 重建跳表指针
            self.delta_index._build_skip_pointers()
            
            print(f"\n跳表步长: {step}")
            
            # 估计存储开销（跳表指针数量）
            total_pointers = 0
            for term, head in self.delta_index.index.items():
                current = head.next
                count = 0
                while current is not None:
                    if count % step == 0:
                        total_pointers += 1
                    current = current.next
                    count += 1
            
            # 测量检索时间
            start_time = time.time()
            results = self.boolean_search(self.delta_index, query)
            search_time = time.time() - start_time
            
            performance_data.append({
                'step': step,
                'pointers': total_pointers,
                'time': search_time,
                'results': len(results)
            })
            
            print(f"  估计跳表指针数量: {total_pointers}")
            print(f"  检索结果数: {len(results)}")
            print(f"  检索时间: {search_time*1000:.2f}毫秒")
        
        # 恢复原始步长
        self.delta_index.skip_list_factor = original_step
        self.delta_index._build_skip_pointers()
        
        # 分析结果
        print("\n跳表步长性能分析总结:")
        print("- 存储影响: 步长越小，需要的跳表指针越多，存储开销越大")
        print("- 检索影响: 步长在适中范围内(4-8)时，检索性能最佳")
        print("- 权衡建议: 在实际应用中，通常选择4-8的步长作为默认值")
        
        # 找出最佳步长
        if performance_data:
            best_performance = min(performance_data, key=lambda x: x['time'])
            print(f"- 在当前测试中，最佳步长为 {best_performance['step']}，检索时间 {best_performance['time']*1000:.2f}毫秒")
    
    # ---------------------- 向量空间模型任务 ----------------------
    def task_b_vector_space_model(self):
        """任务B: 向量空间模型检索"""
        print("\n=== 任务B: 向量空间模型检索 ===")
        
        # 设计查询条件
        queries = [
            "boston atheist meetup",
            "computer security internet privacy",
            "technology exploration monthly meetings"
        ]
        
        for query in queries:
            print(f"\n查询: '{query}'")
            
            # 使用差值编码索引进行向量空间检索
            start_time = time.time()
            results = self.vector_space_search(query, self.delta_index, top_k=3)
            search_time = time.time() - start_time
            
            print(f"  检索结果数: {len(results)}")
            print(f"  耗时: {search_time*1000:.2f}毫秒")
            
            if results:
                print(f"  Top-3 结果:")
                for i, res in enumerate(results, 1):
                    print(f"    {i}. {res['doc_id']}: {res['doc_name']} (相似度: {res['score']:.4f})")
                    
                    # 分析TF-IDF值
                    print("      词项TF-IDF分析:")
                    query_terms = query.lower().split()
                    total_docs = len(self.delta_index.documents)
                    
                    for term in query_terms:
                        # 计算文档中该词项的TF-IDF
                        freq = 0
                        term_results = self.delta_index.search_term(term)
                        for result in term_results:
                            if result['doc_id'] == res['doc_id']:
                                freq = result['frequency']
                                break
                        
                        if freq > 0:
                            doc_tf = freq / self.delta_index.document_lengths.get(res['doc_id'], 1)
                            doc_freq = self.delta_index.document_frequencies.get(term, 0)
                            doc_idf = math.log(total_docs / (doc_freq + 1))
                            doc_tfidf = doc_tf * doc_idf
                            print(f"        - {term}: TF={doc_tf:.4f}, IDF={doc_idf:.4f}, TF-IDF={doc_tfidf:.4f}")
    
    def vector_space_search(self, query, index=None, top_k=10):
        """基于向量空间模型的检索"""
        if index is None:
            index = self.delta_index
        
        if not query or not index.index:
            return []
        
        # 分词查询
        query_terms = query.lower().split()
        
        # 计算文档得分
        doc_scores = defaultdict(float)
        query_vector = {}
        
        # 计算查询向量的TF-IDF
        total_docs = len(index.documents)
        for term in query_terms:
            if term not in query_vector:
                doc_freq = index.document_frequencies.get(term, 0)
                idf = math.log(total_docs / (doc_freq + 1)) if doc_freq > 0 else 0
                # 查询的TF是词项在查询中出现的次数
                tf = query_terms.count(term) / len(query_terms)
                query_vector[term] = tf * idf
        
        # 计算文档向量与查询向量的相似度
        for doc_id in index.documents:
            # 计算文档向量和查询向量的点积
            dot_product = 0
            doc_norm = 0
            query_norm = 0
            
            # 计算文档向量的TF-IDF和模长
            for term in query_terms:
                # 查找词项在文档中的频率
                freq = 0
                term_results = index.search_term(term)
                for result in term_results:
                    if result['doc_id'] == doc_id:
                        freq = result['frequency']
                        break
                
                # 计算文档的TF-IDF
                if freq > 0:
                    doc_tf = freq / index.document_lengths.get(doc_id, 1)
                    doc_idf = math.log(total_docs / (index.document_frequencies.get(term, 0) + 1))
                    doc_tfidf = doc_tf * doc_idf
                    doc_norm += doc_tfidf ** 2
                    dot_product += doc_tfidf * query_vector.get(term, 0)
            
            # 计算查询向量的模长
            for term in query_vector:
                query_norm += query_vector[term] ** 2
            
            # 计算余弦相似度
            if doc_norm > 0 and query_norm > 0:
                similarity = dot_product / (math.sqrt(doc_norm) * math.sqrt(query_norm))
                doc_scores[doc_id] = similarity
        
        # 按相似度排序并返回前top_k个结果
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # 转换为文档信息列表
        results = []
        for doc_id, score in sorted_docs:
            if doc_id in index.documents:
                results.append({
                    'doc_id': doc_id,
                    'doc_name': index.documents[doc_id].doc_name,
                    'score': score
                })
        
        return results
    
    def run_all_tasks(self):
        """运行所有测试任务"""
        # 布尔检索任务
        self.task_a1_complex_queries()
        self.task_a2_compression_efficiency()
        self.task_a3_phrase_search()
        self.task_a4_skip_list_analysis()
        
        # 向量空间模型任务
        self.task_b_vector_space_model()

if __name__ == "__main__":
    # 创建测试实例
    tests = OptimizedRetrievalTests()
    
    # 运行所有测试任务
    tests.run_all_tasks()