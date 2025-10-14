import os
import json
from datetime import datetime
from typing import List, Dict
from .parser import FixedDocumentParser
from .tokenizer import TextTokenizer
from .normalizer import TextNormalizer

class DocumentGenerator:
    def __init__(self, output_dir="documents"):
        self.parser = FixedDocumentParser()
        self.tokenizer = TextTokenizer()
        self.normalizer = TextNormalizer()
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "raw_documents"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "tokenized_documents"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "normalized_documents"), exist_ok=True)
    
    def generate_document_from_file(self, file_path: str) -> Dict:
        """从单个文件生成完整的检索文档"""
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        
        print(f"\n{'='*60}")
        print(f"Generating document from: {filename}")
        print(f"{'='*60}")
        
        # 1. 解析原始内容
        if file_path.endswith('.xml'):
            raw_content = self.parser.parse_meetup_xml(file_path)
            if not raw_content:
                raw_content = self.parser.parse_xml_file(file_path)
        elif file_path.endswith('.json'):
            raw_content = self.parser.parse_json_file(file_path)
        else:
            print(f"Unsupported file format: {file_path}")
            return {}
        
        if not raw_content:
            print("No content extracted from file")
            return {}
        
        # 2. 分词处理
        tokens = self.tokenizer.advanced_tokenize(raw_content)
        
        # 3. 规范化处理
        normalized_tokens = self.normalizer.normalize_tokens(tokens)
        
        # 4. 生成各种版本的文档
        document_info = self._create_document_versions(
            base_name, raw_content, tokens, normalized_tokens
        )
        
        # 5. 生成元数据
        self._generate_metadata(base_name, document_info)
        
        print(f"✓ Document generation completed!")
        print(f"  - Raw document: {document_info['raw_doc_path']}")
        print(f"  - Tokenized document: {document_info['tokenized_doc_path']}")
        print(f"  - Normalized document: {document_info['normalized_doc_path']}")
        
        return document_info
    
    def _create_document_versions(self, base_name: str, raw_content: str, 
                                tokens: List[str], normalized_tokens: List[str]) -> Dict:
        """创建不同版本的文档"""
        
        # 原始文档（解析后的完整文本）
        raw_doc_path = os.path.join(self.output_dir, "raw_documents", f"{base_name}_raw.txt")
        with open(raw_doc_path, 'w', encoding='utf-8') as f:
            f.write(raw_content)
        
        # 分词后的文档（保留原始单词）
        tokenized_doc_path = os.path.join(self.output_dir, "tokenized_documents", f"{base_name}_tokenized.txt")
        with open(tokenized_doc_path, 'w', encoding='utf-8') as f:
            f.write("TOKENS:\n")
            f.write("=" * 50 + "\n")
            for i, token in enumerate(tokens, 1):
                f.write(f"{i:4d}. {token}\n")
        
        # 规范化后的文档（用于检索的最终版本）
        normalized_doc_path = os.path.join(self.output_dir, "normalized_documents", f"{base_name}_normalized.txt")
        with open(normalized_doc_path, 'w', encoding='utf-8') as f:
            f.write("NORMALIZED TOKENS (For Retrieval):\n")
            f.write("=" * 50 + "\n")
            for i, token in enumerate(normalized_tokens, 1):
                f.write(f"{i:4d}. {token}\n")
            
            # 添加统计信息
            f.write("\n" + "=" * 50 + "\n")
            f.write("STATISTICS:\n")
            f.write(f"Total tokens: {len(normalized_tokens)}\n")
            f.write(f"Unique tokens: {len(set(normalized_tokens))}\n")
        
        # 生成检索用的文档（空格分隔的规范化tokens）
        retrieval_doc_path = os.path.join(self.output_dir, f"{base_name}_retrieval.txt")
        with open(retrieval_doc_path, 'w', encoding='utf-8') as f:
            f.write(" ".join(normalized_tokens))
        
        return {
            'base_name': base_name,
            'raw_content': raw_content,
            'tokens': tokens,
            'normalized_tokens': normalized_tokens,
            'raw_doc_path': raw_doc_path,
            'tokenized_doc_path': tokenized_doc_path,
            'normalized_doc_path': normalized_doc_path,
            'retrieval_doc_path': retrieval_doc_path,
            'token_count': len(tokens),
            'normalized_count': len(normalized_tokens),
            'unique_tokens': len(set(normalized_tokens))
        }
    
    def _generate_metadata(self, base_name: str, doc_info: Dict):
        """生成文档元数据"""
        metadata_path = os.path.join(self.output_dir, f"{base_name}_metadata.json")
        
        metadata = {
            'document_id': base_name,
            'generated_at': datetime.now().isoformat(),
            'processing_steps': {
                'raw_content_length': len(doc_info['raw_content']),
                'token_count': doc_info['token_count'],
                'normalized_token_count': doc_info['normalized_count'],
                'unique_token_count': doc_info['unique_tokens']
            },
            'file_paths': {
                'raw_document': doc_info['raw_doc_path'],
                'tokenized_document': doc_info['tokenized_doc_path'],
                'normalized_document': doc_info['normalized_doc_path'],
                'retrieval_document': doc_info['retrieval_doc_path']
            },
            'sample_content': {
                'first_100_chars': doc_info['raw_content'][:100] + "..." if len(doc_info['raw_content']) > 100 else doc_info['raw_content'],
                'first_20_tokens': doc_info['tokens'][:20],
                'first_20_normalized': doc_info['normalized_tokens'][:20]
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def generate_documents_from_directory(self, input_dir: str) -> Dict[str, Dict]:
        """从目录中的所有文件生成文档"""
        all_documents = {}
        
        if not os.path.exists(input_dir):
            print(f"Input directory not found: {input_dir}")
            return all_documents
        
        # 查找所有支持的文件
        supported_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.xml', '.json')):
                    supported_files.append(os.path.join(root, file))
        
        print(f"Found {len(supported_files)} files to process")
        
        for i, file_path in enumerate(supported_files, 1):
            print(f"\n[{i}/{len(supported_files)}] Processing: {os.path.basename(file_path)}")
            
            doc_info = self.generate_document_from_file(file_path)
            if doc_info:
                all_documents[file_path] = doc_info
        
        # 生成总体统计
        self._generate_global_statistics(all_documents)
        
        return all_documents
    
    def _generate_global_statistics(self, all_documents: Dict[str, Dict]):
        """生成全局统计信息"""
        if not all_documents:
            return
        
        total_stats = {
            'total_documents': len(all_documents),
            'total_raw_characters': 0,
            'total_tokens': 0,
            'total_normalized_tokens': 0,
            'total_unique_tokens': 0,
            'average_tokens_per_document': 0,
            'documents': {}
        }
        
        for file_path, doc_info in all_documents.items():
            filename = os.path.basename(file_path)
            total_stats['total_raw_characters'] += len(doc_info['raw_content'])
            total_stats['total_tokens'] += doc_info['token_count']
            total_stats['total_normalized_tokens'] += doc_info['normalized_count']
            total_stats['total_unique_tokens'] += doc_info['unique_tokens']
            
            total_stats['documents'][filename] = {
                'token_count': doc_info['token_count'],
                'normalized_count': doc_info['normalized_count'],
                'unique_tokens': doc_info['unique_tokens']
            }
        
        total_stats['average_tokens_per_document'] = total_stats['total_normalized_tokens'] / len(all_documents)
        
        # 保存全局统计
        stats_path = os.path.join(self.output_dir, "global_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(total_stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print("GLOBAL STATISTICS")
        print(f"{'='*60}")
        print(f"Total documents processed: {total_stats['total_documents']}")
        print(f"Total normalized tokens: {total_stats['total_normalized_tokens']}")
        print(f"Total unique tokens: {total_stats['total_unique_tokens']}")
        print(f"Average tokens per document: {total_stats['average_tokens_per_document']:.1f}")
        print(f"Global statistics saved to: {stats_path}")