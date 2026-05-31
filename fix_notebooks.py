import json

# Only target the file sitting directly next to this script
file_name = 'Kaggle Credit Card Fraud.ipynb'

try:
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Force the notebook format back to v4.4 (GitHub's stable target)
    data['nbformat'] = 4
    data['nbformat_minor'] = 4
    
    # Loop through every cell and completely delete the 'id' field
    cleaned_cells = []
    for cell in data.get('cells', []):
        if 'id' in cell:
            del cell['id']
        cleaned_cells.append(cell)
        
    data['cells'] = cleaned_cells
    
    # Save the completely scrubbed file back down
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        
    print(f"✅ Successfully stripped cell IDs from {file_name}!")
except FileNotFoundError:
    print(f"❌ Error: Python still can't see '{file_name}'. Make sure your terminal path matches your folder path.")
except Exception as e:
    print(f"❌ Unexpected error processing {file_name}: {e}")

input("\nPress Enter to exit...")