from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

# Load your dataset (replace with your dataset path)
data = pd.read_csv('url_dataset.csv')

# Separate legitimate and phishing data
legit_data = data[data['type'] == 'legitimate']
phish_data = data[data['type'] == 'phishing']

# Print initial counts
print(f"Initial counts - Legitimate: {len(legit_data)}, Phishing: {len(phish_data)}")

# Extract domains for legitimate URLs
legit_data['domain'] = legit_data['url'].apply(lambda x: x.split('/')[2])

# Prioritize diverse domains for legitimate URLs
unique_domains = legit_data.drop_duplicates(subset='domain')

# If the unique domains are more than phishing samples, undersample them
phish_count = len(phish_data)
legit_count = int(phish_count * 1.015)  # 1.5% higher than phishing

if len(unique_domains) > legit_count:
    legit_data = unique_domains.sample(legit_count, random_state=42)
else:
    legit_data = unique_domains

# Combine with phishing data
balanced_data = pd.concat([legit_data, phish_data]).sample(frac=1, random_state=42).reset_index(drop=True)

# Print final counts
print(f"Final counts - Legitimate: {balanced_data[balanced_data['type'] == 'legitimate'].shape[0]}, Phishing: {balanced_data[balanced_data['type'] == 'phishing'].shape[0]}")

# Save the balanced dataset
balanced_data.drop(columns=['domain'], inplace=True)
balanced_data.to_csv('balanced_dataset.csv', index=False)

print('Dataset balanced successfully and saved as balanced_dataset.csv')
