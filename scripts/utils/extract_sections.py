from bs4 import BeautifulSoup
import glob
import os
import json

# Path to your TEI files
tei_input_dir = r"C:\Users\W11\rag_project\data\processed\tei"
# Directory where you want to write the JSON outputs
json_output_dir = r"C:\Users\W11\rag_project\data\parsed"
os.makedirs(json_output_dir, exist_ok=True)

def extract_sections(tei_file):
    with open(tei_file, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "xml")

    # Title: take the document title from <title> in the header
    title = soup.find("title").get_text(" ", strip=True) if soup.find("title") else ""

    # Abstract: take all text inside the <abstract> tag
    abstract = ""
    abstract_tag = soup.find("abstract")
    if abstract_tag:
        abstract = abstract_tag.get_text(" ", strip=True)

    # Full body text: flatten all text in <body>
    body_text = ""
    body = soup.find("body")
    if body:
        body_text = body.get_text(" ", strip=True)

    # Introduction: look for a <div> in the body whose <head> contains 'introduction'
    intro_text = ""
    if body:
        # Find all top-level divs inside body
        for div in body.find_all("div", recursive=False):
            head = div.find("head")
            if head and "intro" in head.get_text().lower():
                intro_text = div.get_text(" ", strip=True)
                break
        # Fallback: use the first <div> if no explicit introduction is found
        if not intro_text:
            first_div = body.find("div")
            if first_div:
                intro_text = first_div.get_text(" ", strip=True)

    return {
        "title": title,
        "abstract": abstract,
        "introduction": intro_text,
        "body": body_text,
    }

# Process all TEI files in the input directory
for tei_path in glob.glob(os.path.join(tei_input_dir, "*.tei.xml")):
    data = extract_sections(tei_path)
    base_name = os.path.basename(tei_path).replace(".tei.xml", ".json")
    out_path = os.path.join(json_output_dir, base_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Extracted {base_name}")
