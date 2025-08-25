from langchain_core.tools import tool
import requests
import xml.etree.ElementTree as ET
import os
from dotenv import load_dotenv

load_dotenv()
WOLFRAM_ALPHA_APPID = os.getenv("WOLFRAM_ALPHA_APPID")

@tool
def wolfram(query: str) -> str:
    """Queries Wolframalpha API to get information on math and science topics"""
    url = "http://api.wolframalpha.com/v2/query"
    params = {
        "input": query, 
        "appid": WOLFRAM_ALPHA_APPID, 
        "format": "plaintext"
    }
    
    resp = requests.get(url, params=params)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    results = {}
    for pod in root.findall(".//pod"):
        title = pod.attrib.get("title")
        subpods = []
        for sub in pod.findall("subpod"):
            plain = sub.find("plaintext")
            if plain is not None and plain.text:
                subpods.append(plain.text.strip())
        if subpods:
            results[title] = subpods if len(subpods) > 1 else subpods[0]
    return str(results)



