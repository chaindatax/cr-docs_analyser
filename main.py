from dotenv import load_dotenv

from docs_analyser.azure_analyser import AzureAnalyser
from docs_analyser.mistral_analyser import MistralAnalyser

if __name__ == "__main__":
    load_dotenv()

    file_path = "dataset/id_cards/0b43b0c-frenchID-3907357441.jpg"

    print(MistralAnalyser().runner(file_path))
    print(AzureAnalyser().runner(file_path))