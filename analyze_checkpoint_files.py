import json
import os
from collections import defaultdict
import tqdm
def load_checkpoint_data(file_path):
    """Robustly load multi-line JSON objects from a file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        buffer = []
        brace_count = 0
        for line in tqdm.tqdm(f):
            if '{\n' ==line:
                brace_count += line.count('{')
            if '}\n' == line:
                brace_count -= line.count('}')
            buffer.append(line)

            if brace_count == 0 and buffer:
                block = ''.join(buffer).strip()
                try:
                    #print(f"Loading block: {block}")
                    obj = json.loads(block)
                    data.append(obj)
                except Exception as e:
                    print(f"Skipping malformed block: {e}")
                buffer = []
    print(f"Successfully loaded {len(data)} samples")
    return data

def analyze_checkpoint_file(filepath):
    """Analyze a checkpoint file using the robust loader"""
    if not os.path.exists(filepath):
        return None
    
    print(f"\n=== Analyzing {filepath} ===")
    
    # Get file size
    file_size = os.path.getsize(filepath)
    print(f"File size: {file_size / (1024*1024):.2f} MB")
    
    try:
        # Load all data using the robust function
        records = load_checkpoint_data(filepath)
        
        if not records:
            print("No records loaded successfully")
            return None
        
        # Analyze field structure
        field_counts = defaultdict(int)
        for record in records:
            for key in record.keys():
                field_counts[key] += 1
        
        print(f"Common fields: {list(field_counts.keys())}")
        
        # Show structure of first record
        first_record = records[0]
        print("\nFirst record structure:")
        for key, value in first_record.items():
            if isinstance(value, str):
                preview = value[:100] + "..." if len(value) > 100 else value
                print(f"  {key}: {type(value).__name__} ({len(value)} chars) - {repr(preview)}")
            elif isinstance(value, (list, dict)):
                print(f"  {key}: {type(value).__name__} (len={len(value)})")
                if isinstance(value, dict) and len(value) < 10:
                    for subkey in value.keys():
                        print(f"    - {subkey}")
                elif isinstance(value, list) and len(value) > 0:
                    print(f"    - Sample items: {value[:3]}...")
            else:
                print(f"  {key}: {type(value).__name__} - {value}")
        
        return {
            'file_size': file_size,
            'total_objects': len(records),
            'fields': list(field_counts.keys()),
            'records': records[:5],  # Store first 5 records for website
            'field_counts': dict(field_counts)
        }
        
    except Exception as e:
        print(f"Error analyzing file: {e}")
        return None

def main():
    checkpoint_dir = "output/ckpt"
    
    if not os.path.exists(checkpoint_dir):
        print(f"Directory {checkpoint_dir} not found")
        return
    
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.jsonl')]
    print(f"Found {len(files)} JSONL files in {checkpoint_dir}")
    
    analysis_results = {}
    
    for filename in sorted(files):
        filepath = os.path.join(checkpoint_dir, filename)
        result = analyze_checkpoint_file(filepath)
        if result:
            analysis_results[filename] = result
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    for filename, data in analysis_results.items():
        print(f"\n{filename}:")
        print(f"  Size: {data['file_size'] / (1024*1024):.2f} MB")
        print(f"  Total Objects: {data['total_objects']}")
        print(f"  Fields: {', '.join(data['fields'])}")
    
    return analysis_results

if __name__ == "__main__":
    main() 