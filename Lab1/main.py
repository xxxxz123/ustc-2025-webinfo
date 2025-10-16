import os
import json
import nltk
nltk.download('punkt_tab')
from utils.parser import DataLoader
from utils.tokenizer import TextTokenizer
from utils.normalizer import TextNormalizer

def debug_xml_files():
    """调试XML文件结构"""
    print("\n正在分析XML文件结构...")
    xml_files = DataLoader.get_available_files()
    
    for xml_file in xml_files:
        DataLoader.debug_xml_structure(os.path.join('data', xml_file))

def process_documents_basic(documents, use_lemmatization=True):
    """基础处理文档集合"""
    tokenizer = TextTokenizer()
    normalizer = TextNormalizer(use_lemmatization=use_lemmatization)
    
    processed_documents = []
    
    for i, doc in enumerate(documents):
        print(f"处理文档 {i+1}: {doc['id']} ({doc['source_file']})")
        
        original_text = doc['content']
        
        # 分词
        tokens = tokenizer.tokenize(original_text)
        
        # 规范化
        normalized_tokens = normalizer.normalize(tokens)
        
        processed_doc = {
            'id': doc['id'],
            'name': doc['name'],
            'source_file': doc['source_file'],
            'city': doc.get('city', ''),
            'state': doc.get('state', ''),
            'original_content': original_text,
            'tokens': tokens,  # 分词后的结果
            'normalized_tokens': normalized_tokens,
            'normalized_text': ' '.join(normalized_tokens),
            'phrases': []  # 空列表保持结构一致
        }
        
        processed_documents.append(processed_doc)
        
        print(f"  原始文本长度: {len(original_text)}")
        print(f"  分词后词项数量: {len(tokens)}")
        print(f"  规范化后词项数量: {len(normalized_tokens)}")
        print(f"  规范化示例: {' '.join(normalized_tokens[:8])}...")
    
    return processed_documents

def process_documents_advanced(documents, use_lemmatization=True, phrase_method='adaptive'):
    """使用高级方法处理文档（包含词组识别）"""
    
    if phrase_method == 'external':
        # 使用外部工具
        print("使用外部工具进行词组提取...")
        try:
            from utils.phrase_extractor import PhraseExtractor
            extractor = PhraseExtractor()
        except ImportError:
            print("警告: 外部工具未安装，回退到自适应方法")
            return process_documents_advanced(documents, use_lemmatization, 'adaptive')
        
        processed_docs = []
        for doc in documents:
            text = doc['content']
            
            # 使用外部工具提取短语
            external_phrases = extractor.ensemble_extraction(text)
            print(f"  文档 {doc['id']} 检测到 {len(external_phrases)} 个外部词组")
            
            # 基础分词
            tokenizer = TextTokenizer(enable_phrase_detection=True)
            all_texts = [text]
            base_tokens, _ = tokenizer.tokenize_with_phrases(all_texts)
            base_tokens = base_tokens[0] if base_tokens else []
            
            # 合并外部工具检测的短语
            final_tokens = []
            for token in base_tokens:
                if '_' in token:
                    final_tokens.append(token)
                else:
                    # 检查是否属于外部检测的短语
                    found = False
                    for phrase in external_phrases:
                        if token in phrase.replace('_', ' ').split():
                            found = True
                            break
                    if not found:
                        final_tokens.append(token)
            
            # 添加外部检测的新短语
            for phrase in external_phrases:
                if phrase not in ' '.join(final_tokens):
                    final_tokens.append(phrase)
            
            # 规范化处理
            normalizer = TextNormalizer(use_lemmatization=use_lemmatization)
            normalized_tokens = normalizer.normalize(final_tokens)
            
            processed_docs.append({
                'id': doc['id'],
                'name': doc['name'],
                'source_file': doc['source_file'],
                'city': doc.get('city', ''),
                'state': doc.get('state', ''),
                'original_content': text,
                'tokens': final_tokens,
                'normalized_tokens': normalized_tokens,
                'normalized_text': ' '.join(normalized_tokens),
                'phrases': list(external_phrases),
                'phrase_method': 'external'
            })
        
        return processed_docs
    
    else:
        # 使用自适应方法
        print("使用自适应词组检测方法...")
        tokenizer = TextTokenizer(enable_phrase_detection=True)
        all_texts = [doc['content'] for doc in documents]
        all_tokens, learned_phrases = tokenizer.tokenize_with_phrases(all_texts)
        
        print(f"从所有文档中学习到 {len(learned_phrases)} 个常用词组")
        
        normalizer = TextNormalizer(use_lemmatization=use_lemmatization)
        processed_docs = []
        
        for i, (doc, tokens) in enumerate(zip(documents, all_tokens)):
            normalized_tokens = normalizer.normalize(tokens)
            
            # 提取当前文档中出现的词组
            doc_phrases = [p for p in learned_phrases if p in tokens]
            
            processed_docs.append({
                'id': doc['id'],
                'name': doc['name'],
                'source_file': doc['source_file'],
                'city': doc.get('city', ''),
                'state': doc.get('state', ''),
                'original_content': doc['content'],
                'tokens': tokens,
                'normalized_tokens': normalized_tokens,
                'normalized_text': ' '.join(normalized_tokens),
                'phrases': doc_phrases,
                'phrase_method': 'adaptive'
            })
            
            print(f"处理文档 {i+1}: {doc['id']} ({doc['source_file']})")
            print(f"  原始文本长度: {len(doc['content'])}")
            print(f"  分词后词项数量: {len(tokens)}")
            print(f"  规范化后词项数量: {len(normalized_tokens)}")
            print(f"  检测到词组: {len(doc_phrases)} 个")
            print(f"  分词示例: {' '.join(tokens[:8])}...")
            print(f"  规范化示例: {' '.join(normalized_tokens[:8])}...")
        
        return processed_docs

def process_documents(documents, use_lemmatization=True, processing_mode='basic'):
    """统一的文档处理函数"""
    if processing_mode == 'advanced_adaptive':
        return process_documents_advanced(documents, use_lemmatization, phrase_method='adaptive')
    elif processing_mode == 'advanced_external':
        return process_documents_advanced(documents, use_lemmatization, phrase_method='external')
    else:
        return process_documents_basic(documents, use_lemmatization)

def save_documents_in_structure(processed_documents, output_base_dir='generated_documents'):
    """按照指定结构保存文档"""
    
    # 创建目录结构
    directories = {
        'raw': os.path.join(output_base_dir, 'raw_documents'),
        'tokenized': os.path.join(output_base_dir, 'tokenized_documents'),
        'normalized': os.path.join(output_base_dir, 'normalized_documents')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    retrieval_file = os.path.join(output_base_dir, 'sample_event_retrieval.txt')
    
    # 保存各个阶段的文档
    for doc in processed_documents:
        # 创建安全的文件名
        safe_name = f"doc_{doc['id']}_{doc['name']}"
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')[:50]
        
        # 1. 原始文档
        raw_file = os.path.join(directories['raw'], f"{safe_name}_raw.txt")
        with open(raw_file, 'w', encoding='utf-8') as f:
            f.write(f"事件ID: {doc['id']}\n")
            f.write(f"事件名称: {doc['name']}\n")
            f.write(f"来源文件: {doc['source_file']}\n")
            if doc['city']:
                f.write(f"地点: {doc['city']}, {doc['state']}\n")
            f.write(f"\n原始内容:\n{doc['original_content']}\n")
        
        # 2. 分词后文档
        tokenized_file = os.path.join(directories['tokenized'], f"{safe_name}_tokenized.txt")
        with open(tokenized_file, 'w', encoding='utf-8') as f:
            f.write(f"事件ID: {doc['id']}\n")
            f.write(f"事件名称: {doc['name']}\n")
            f.write(f"来源文件: {doc['source_file']}\n")
            if doc.get('phrases'):
                f.write(f"检测到的常用词组: {', '.join(doc['phrases'])}\n")
            if doc.get('phrase_method'):
                f.write(f"词组检测方法: {doc['phrase_method']}\n")
            f.write(f"\n分词结果 (共{len(doc['tokens'])}个词项):\n")
            f.write(' '.join(doc['tokens']) + '\n')
            f.write(f"\n详细词项列表:\n")
            for i, token in enumerate(doc['tokens'], 1):
                marker = " [PHRASE]" if '_' in token else ""
                f.write(f"{i:3d}. {token}{marker}\n")
        
        # 3. 规范化后文档
        normalized_file = os.path.join(directories['normalized'], f"{safe_name}_normalized.txt")
        with open(normalized_file, 'w', encoding='utf-8') as f:
            f.write(f"事件ID: {doc['id']}\n")
            f.write(f"事件名称: {doc['name']}\n")
            f.write(f"来源文件: {doc['source_file']}\n")
            if doc.get('phrases'):
                f.write(f"检测到的常用词组: {', '.join(doc['phrases'])}\n")
            f.write(f"\n规范化结果 (共{len(doc['normalized_tokens'])}个词项):\n")
            f.write(' '.join(doc['normalized_tokens']) + '\n')
            f.write(f"\n详细词项列表:\n")
            for i, token in enumerate(doc['normalized_tokens'], 1):
                marker = " [PHRASE]" if '_' in token else ""
                f.write(f"{i:3d}. {token}{marker}\n")
    
    # 4. 检索用文档（空格分隔的规范化文本）
    with open(retrieval_file, 'w', encoding='utf-8') as f:
        f.write("# 检索用文档 - 空格分隔的规范化文本\n")
        f.write("# 每行代表一个文档，词项之间用空格分隔\n")
        f.write("# 格式: 文档ID | 文档名称 | 规范化文本\n\n")
        
        for doc in processed_documents:
            retrieval_line = f"{doc['id']} | {doc['name']} | {doc['normalized_text']}"
            f.write(retrieval_line + '\n')
    
    # 生成统计信息文件
    stats_file = os.path.join(output_base_dir, 'processing_statistics.txt')
    total_raw_tokens = sum(len(doc['tokens']) for doc in processed_documents)
    total_normalized_tokens = sum(len(doc['normalized_tokens']) for doc in processed_documents)
    total_phrases = sum(len(doc.get('phrases', [])) for doc in processed_documents)
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=== 文档处理统计 ===\n\n")
        f.write(f"总文档数: {len(processed_documents)}\n")
        f.write(f"总原始词项数: {total_raw_tokens}\n")
        f.write(f"总规范化词项数: {total_normalized_tokens}\n")
        f.write(f"检测到的常用词组总数: {total_phrases}\n")
        f.write(f"平均每文档原始词项数: {total_raw_tokens/len(processed_documents):.1f}\n")
        f.write(f"平均每文档规范化词项数: {total_normalized_tokens/len(processed_documents):.1f}\n")
        f.write(f"平均每文档词组数: {total_phrases/len(processed_documents):.1f}\n\n")
        
        # 词组统计
        if total_phrases > 0:
            all_phrases = []
            for doc in processed_documents:
                all_phrases.extend(doc.get('phrases', []))
            phrase_freq = {}
            for phrase in all_phrases:
                phrase_freq[phrase] = phrase_freq.get(phrase, 0) + 1
            
            f.write("常用词组频率统计:\n")
            for phrase, freq in sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)[:20]:
                f.write(f"  {phrase}: {freq} 次\n")
            f.write("\n")
        
        f.write("目录结构:\n")
        f.write(f"generated_documents/\n")
        f.write(f"├── raw_documents/              # 原始解析内容 ({len(processed_documents)} 个文件)\n")
        f.write(f"├── tokenized_documents/        # 分词后的文档 ({len(processed_documents)} 个文件)\n")
        f.write(f"├── normalized_documents/       # 规范化后的文档 ({len(processed_documents)} 个文件)\n")
        f.write(f"├── sample_event_retrieval.txt  # 检索用文档 (1 个文件)\n")
        f.write(f"└── processing_statistics.txt   # 统计信息\n\n")
        
        f.write("文件列表:\n")
        for doc in processed_documents:
            safe_name = f"doc_{doc['id']}_{doc['name'].replace(' ', '_')[:30]}"
            phrase_count = len(doc.get('phrases', []))
            f.write(f"- {safe_name}: {len(doc['normalized_tokens'])} 个规范化词项, {phrase_count} 个词组\n")
    
    print(f"\n文档已按照指定结构保存到: {output_base_dir}/")
    print(f"目录结构:")
    print(f"  {output_base_dir}/")
    print(f"  ├── raw_documents/              # 原始解析内容")
    print(f"  ├── tokenized_documents/        # 分词后的文档") 
    print(f"  ├── normalized_documents/       # 规范化后的文档")
    print(f"  ├── sample_event_retrieval.txt  # 检索用文档")
    print(f"  └── processing_statistics.txt   # 统计信息")
    
    return {
        'base_dir': output_base_dir,
        'raw_dir': directories['raw'],
        'tokenized_dir': directories['tokenized'],
        'normalized_dir': directories['normalized'],
        'retrieval_file': retrieval_file,
        'stats_file': stats_file
    }

def main():
    nltk.download('averaged_perceptron_tagger_eng')
    """主函数"""
    print("Meetup文本处理系统")
    print("=" * 50)
    
    try:
        debug_xml_files()
        
        # 加载所有XML文件
        print("\n正在加载data目录下的所有XML文件...")
        documents = DataLoader.load_all_xml_files()
        
        if not documents:
            print("没有找到可处理的事件数据")
            print("请检查XML文件格式，或运行调试模式查看文件结构")
            return
        
        # 选择处理模式
        print(f"\n找到 {len(documents)} 个事件")
        print("\n请选择分词处理模式:")
        print("1. 基础模式 (快速处理)")
        print("2. 高级模式-自适应词组检测")
        print("3. 高级模式-外部工具词组检测(不建议，所需工具在requirements.txt中)")
        
        mode_choice = input("请选择处理模式 (默认1): ").strip()
        
        if mode_choice == "2":
            processing_mode = 'advanced_adaptive'
            mode_name = "高级模式-自适应词组检测"
        elif mode_choice == "3":
            processing_mode = 'advanced_external'
            mode_name = "高级模式-外部工具词组检测"
        else:
            processing_mode = 'basic'
            mode_name = "基础模式"
        
        # 选择规范化方式
        norm_choice = input("使用词形还原(1)还是词干提取(2)? (默认1): ").strip()
        use_lemmatization = norm_choice != "2"
        
        print(f"\n开始处理文档...")
        print(f"  处理模式: {mode_name}")
        print(f"  规范化方式: {'词形还原' if use_lemmatization else '词干提取'}")
        
        # 处理文档
        processed_docs = process_documents(documents, use_lemmatization, processing_mode)
        
        # 按照指定结构保存文档
        output_structure = save_documents_in_structure(processed_docs)
        
        print("\n处理完成!")
        print(f"\n生成的文档结构:")
        print(f"总文档数: {len(processed_docs)}")
        print(f"输出目录: {output_structure['base_dir']}/")
        
        # 显示前几个文档的信息
        print(f"\n前3个文档示例:")
        for i, doc in enumerate(processed_docs[:3], 1):
            print(f"  {i}. {doc['name']} (ID: {doc['id']})")
            print(f"     规范化词项: {len(doc['normalized_tokens'])} 个")
            print(f"     检测到词组: {len(doc.get('phrases', []))} 个")
            print(f"     示例: {' '.join(doc['normalized_tokens'][:5])}...")
            if doc.get('phrases'):
                print(f"     词组示例: {', '.join(doc['phrases'][:3])}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    main()