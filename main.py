import os
import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_latest_10k_filing_url(cik="0001652044"):
    """
    Gets the URL of the latest 10-K filing for a given CIK
    using the SEC's more reliable JSON data feed.
    """
    print("Fetching latest 10-K filing URL from the SEC JSON API...")
    
    # Pad the CIK with leading zeros to make it 10 digits
    padded_cik = cik.zfill(10)
    
    # Construct the URL for the SEC's submissions JSON feed
    json_url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"
    
    # Set a compliant User-Agent
    headers = {'User-Agent': 'Gemini Financial Analysis Project student@example.com'}
    
    try:
        response = requests.get(json_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Access the 'filings' and 'recent' keys
        recent_filings = data['filings']['recent']
        
        # Find the latest 10-K filing
        for i in range(len(recent_filings['form'])):
            if recent_filings['form'][i] == '10-K':
                accession_number = recent_filings['accessionNumber'][i].replace('-', '')
                primary_document = recent_filings['primaryDocument'][i]
                
                # Construct the final URL to the HTML document
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{primary_document}"
                print(f"Successfully found 10-K URL: {filing_url}")
                return filing_url
                
        print("Error: Could not find a 10-K filing in the recent filings.")
        return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching SEC JSON data: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing JSON data. The structure might have changed. Details: {e}")
        return None

def download_and_save_unstructured_data(url, output_path="data/alphabet_10k_unstructured.txt"):
    """Downloads the HTML from the URL, extracts text, and saves it."""
    if not url:
        print("Download skipped as no URL was provided.")
        return False
        
    print(f"Downloading unstructured data from: {url}")
    headers = {'User-Agent': 'Gemini Financial Analysis Project student@example.com'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        text_content = soup.get_text(separator='\n', strip=True)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
            
        print(f"SUCCESS: Unstructured data saved to '{output_path}'")
        return True

    except Exception as e:
        print(f"An error occurred during unstructured data download/processing: {e}")
        return False

# def create_structured_data(output_path="data/alphabet_financials_structured.csv"):
#     """Creates the structured Q&A CSV file."""
#     print("\nCreating structured Q&A data...")
#     data = {
#         'Question': [
#             "What was Alphabet's revenue in 2023?",
#             "What was Alphabet's net income in 2023?",
#             "What was Alphabet's diluted earnings per share (EPS) in 2023?",
#             "What was Alphabet's revenue in 2022?",
#             "What was Alphabet's net income in 2022?",
#             "How many full-time employees did Alphabet have at the end of 2023?"
#         ],
#         'Answer': [
#             "Alphabet's revenue in 2023 was $307.39 billion.",
#             "Alphabet's net income in 2023 was $73.80 billion.",
#             "Alphabet's diluted earnings per share (EPS) in 2023 was $5.80.",
#             "Alphabet's revenue in 2022 was $282.84 billion.",
#             "Alphabet's net income in 2022 was $59.97 billion.",
#             "As of December 31, 2023, Alphabet had 182,502 full-time employees."
#         ]
#     }
    
#     df = pd.DataFrame(data)
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     df.to_csv(output_path, index=False)
    
#     print(f"SUCCESS: Structured Q&A data saved to '{output_path}'")
#     print("\n--- CSV Content Preview ---")
#     print(df.head())
#     print("--------------------------\n")
#     return True



def create_financial_csv(output_path="data/alphabet_financials_structured.csv"):
    """
    Creates a structured CSV file with tabular financial data for Alphabet Inc.,
    perfectly matching the format used in the reference article.
    """
    print("Creating structured tabular data...")
    
    # Data is structured in a tabular format, just like the article's example
    revenue_data = {
        'year': [2023, 2023, 2023, 2023, 2022, 2022, 2022, 2022],
        'quarter': ['Q4', 'Q3', 'Q2', 'Q1', 'Q4', 'Q3', 'Q2', 'Q1'],
        'revenue_usd_billions': [86.31, 76.69, 74.60, 69.79, 76.05, 69.09, 69.69, 68.01],
        'net_income_usd_billions': [20.69, 19.69, 18.37, 15.05, 13.62, 13.91, 16.00, 16.44]
    }
    
    # Create DataFrame from dictionary
    df = pd.DataFrame(revenue_data)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save DataFrame to CSV file
    df.to_csv(output_path, index=False)
    
    print(f"SUCCESS: Structured tabular data has been saved to '{output_path}'")
    print("\n--- CSV Content Preview ---")
    print(df)
    print("--------------------------\n")

if __name__ == "__main__":
    # Step 1: Get the unstructured data (the 10-K report)
    #filing_url = get_latest_10k_filing_url()
    #unstructured_success = download_and_save_unstructured_data(filing_url)
    
    # Step 2: Create the structured data (the financials CSV)
    structured_success = create_financial_csv()
    
    # if unstructured_success and structured_success:
    #     print("\nAll data has been successfully prepared and is ready for the next step.")
    # else:
    #     print("\nThere was an error in the data preparation. Please review the messages above.")

