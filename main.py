
from pprint import pprint

from dotenv import load_dotenv

from docs_analyser.azure_analyser import azure_document_understanding
from docs_analyser.mistral_analyser import mistral_ocr

if __name__ == "__main__":
    load_dotenv()

    pprint(mistral_ocr("dataset/id_cards/0b43b0c-frenchID-3907357441.jpg"))
    pprint(azure_document_understanding("dataset/id_cards/0b43b0c-frenchID-3907357441.jpg"))
    # pprint(main("dataset/passports/vias-pages-3783188661.jpg"))
    # pprint(main("dataset/passports/Ecp5OYUWAAAdVec-2854078686.jpg"))
    # pprint(mistral_ocr("dataset/passports/french-diplomatic-passport-v0-0xptroj0uxia1-1085992610.jpg"))
    # pprint(azure_document_understanding("dataset/passports/french-diplomatic-passport-v0-0xptroj0uxia1-1085992610.jpg"))

