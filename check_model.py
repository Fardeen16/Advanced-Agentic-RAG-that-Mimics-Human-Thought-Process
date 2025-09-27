import google.generativeai as genai
import os

# --- PASTE YOUR API KEY HERE ---
API_KEY = "AIzaSyC64jh0jHd6OegJcXsfMpp__nm4kLBH1dg"

if API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
    print("FATAL: Please replace 'YOUR_GOOGLE_API_KEY_HERE' with your actual Google API key.")
    exit()

try:
    # Configure the client with your API key
    genai.configure(api_key=API_KEY)

    print("--- Models Available to Your API Key ---")
    
    # List all available models
    for m in genai.list_models():
      # Check if the model supports the 'generateContent' method, which is what we need
      if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please ensure your API key is correct and has the 'Generative Language API' enabled in your Google Cloud project.")

