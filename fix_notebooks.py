import nbformat

files = ['Kaggle Credit Card Fraud.ipynb', 'H&M Recommendations.ipynb']

for file_name in files:
    try:
        # Read the notebook forcing standard version 4 compatibility
        with open(file_name, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Overwrite the file, stripping out modern v5 formatting details
        with open(file_name, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f, version=nbformat.NO_CONVERT)
            
        print(f"✅ Successfully updated {file_name}")
    except Exception as e:
        print(f"❌ Error processing {file_name}: {e}")