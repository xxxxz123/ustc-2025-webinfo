import xml.etree.ElementTree as ET
import json
import os
import glob

class MeetupXMLParser:
    def __init__(self, xml_file_path):
        self.xml_file_path = xml_file_path
        try:
            self.tree = ET.parse(xml_file_path)
            self.root = self.tree.getroot()
        except FileNotFoundError:
            print(f"错误: 找不到文件 {xml_file_path}")
            raise
        except ET.ParseError as e:
            print(f"错误: XML文件格式不正确 - {e}")
            raise
    
    def extract_events_with_descriptions(self):
        """提取所有包含description的事件"""
        events_data = []
        
        # 查找所有可能包含事件的元素
        # 首先尝试查找<Event>元素（标准Meetup格式）
        event_elements = self.root.findall('.//Event')
        
        # 如果没有找到<Event>，尝试查找<item>元素（您的XML格式）
        if not event_elements:
            event_elements = self.root.findall('.//item')
        
        # 如果还是没有找到，将根元素本身作为事件处理
        if not event_elements and (self.root.find('description') is not None or 
                                 self.root.find('.//description') is not None):
            event_elements = [self.root]
        
        for event in event_elements:
            event_data = self._extract_event_data(event)
            if event_data and event_data['description']:
                events_data.append(event_data)
        
        return events_data
    
    def _extract_event_data(self, event_element):
        """从事件元素中提取数据"""
        event_data = {}
        
        # 提取ID
        event_id = event_element.find('id')
        if event_id is None:
            # 如果没有id，使用文件名或其他标识符
            event_id = os.path.basename(self.xml_file_path).replace('.xml', '')
        else:
            event_id = event_id.text
        
        # 提取名称
        event_name = event_element.find('name')
        if event_name is None:
            event_name = event_element.find('.//name')  # 尝试深层查找
        
        # 提取描述
        event_description = event_element.find('description')
        if event_description is None:
            event_description = event_element.find('.//description')  # 尝试深层查找
        
        event_data['id'] = str(event_id) if event_id else ''
        event_data['name'] = event_name.text if event_name is not None else 'Unnamed Event'
        event_data['description'] = event_description.text if event_description is not None else ''
        
        # 提取其他可能有用的信息
        event_time = event_element.find('time')
        event_data['time'] = event_time.text if event_time is not None else ''
        
        event_city = event_element.find('city')
        event_data['city'] = event_city.text if event_city is not None else ''
        
        event_state = event_element.find('state')
        event_data['state'] = event_state.text if event_state is not None else ''
        
        return event_data
    
    def merge_events_to_documents(self, events_data):
        """将多个event合并为待检索文档"""
        documents = []
        
        for event in events_data:
            # 将event的各个部分合并为一篇文档
            doc_content = f"{event['name']} {event['description']}"
            documents.append({
                'id': event['id'],
                'name': event['name'],
                'source_file': os.path.basename(self.xml_file_path),
                'city': event.get('city', ''),
                'state': event.get('state', ''),
                'content': doc_content
            })
        
        return documents

class DataLoader:
    @staticmethod
    def load_all_xml_files(data_dir='data'):
        """加载data目录下的所有XML文件"""
        xml_files = glob.glob(os.path.join(data_dir, '*.xml'))
        
        if not xml_files:
            raise FileNotFoundError(f"在目录 {data_dir} 中未找到任何XML文件")
        
        all_documents = []
        total_events = 0
        
        for xml_file in xml_files:
            print(f"正在处理文件: {os.path.basename(xml_file)}")
            try:
                parser = MeetupXMLParser(xml_file)
                events_data = parser.extract_events_with_descriptions()
                documents = parser.merge_events_to_documents(events_data)
                all_documents.extend(documents)
                file_events_count = len(documents)
                total_events += file_events_count
                print(f"  从 {os.path.basename(xml_file)} 提取了 {file_events_count} 个事件")
                
                # 调试信息：显示找到的事件详情
                if file_events_count > 0:
                    for i, event in enumerate(events_data[:2]):  # 显示前2个事件的预览
                        desc_preview = event['description'][:50] + "..." if len(event['description']) > 50 else event['description']
                        print(f"    事件 {i+1}: {event['name']} - {desc_preview}")
                
            except Exception as e:
                print(f"  处理文件 {os.path.basename(xml_file)} 时出错: {e}")
                continue
        
        print(f"\n总计从 {len(xml_files)} 个文件中提取了 {total_events} 个事件")
        return all_documents
    
    @staticmethod
    def get_available_files(data_dir='data'):
        """获取可用的数据文件列表"""
        xml_files = glob.glob(os.path.join(data_dir, '*.xml'))
        return [os.path.basename(f) for f in xml_files]
    
    @staticmethod
    def debug_xml_structure(xml_file_path):
        """调试XML文件结构"""
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            
            print(f"\n调试文件: {os.path.basename(xml_file_path)}")
            print(f"根元素: {root.tag}")
            
            # 查找所有直接子元素
            print("直接子元素:")
            for child in root:
                print(f"  {child.tag}: {child.text[:50] if child.text else 'None'}")
            
            # 查找description元素
            descriptions = root.findall('.//description')
            print(f"找到的描述元素数量: {len(descriptions)}")
            for i, desc in enumerate(descriptions):
                print(f"  描述 {i+1}: {desc.text[:100] if desc.text else 'None'}")
            
            # 查找可能的事件元素
            events = root.findall('.//Event')
            items = root.findall('.//item')
            print(f"找到的Event元素: {len(events)}")
            print(f"找到的item元素: {len(items)}")
            
        except Exception as e:
            print(f"调试时出错: {e}")