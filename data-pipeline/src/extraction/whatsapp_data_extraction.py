import re
import csv
import spacy
import os
from spacy.matcher import PhraseMatcher
from tqdm import tqdm

def extract_housing_listings(input_file):
    """
    Extract and structure housing listings from WhatsApp chat exports.
    
    Parameters:
    -----------
    input_file : str
        Path to the WhatsApp chat .txt file
    
    Returns:
    --------
    None
        Saves two CSV files:
        - relevant_listings.csv: Filtered housing-related messages
        - structured_listings_nlp.csv: Structured data with extracted fields
    """
    output_dir = "./data/raw"
    # Make sure folder exists
    os.makedirs(output_dir, exist_ok=True)
    # Full paths
    output_file = os.path.join(output_dir, "structured_listings_nlp.csv")
    filtered_file = os.path.join(output_dir, "relevant_listings.csv")
    # output_file = "structured_listings_nlp.csv"
    # filtered_file = "relevant_listings.csv"
    
    # print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_trf")
    
    # --- Define phrase lists for semantic matching ---
    requirement_phrases = {
        "looking": ["looking for", "need", "wanted", "in search of", "searching for"],
        "offering": ["available", "offering", "spot open", "for rent", "vacant", "sublet", "accommodation open"]
    }
    
    gender_phrases = {
        "female": ["female", "girl", "girls", "women", "ladies", "only girls"],
        "male": ["male", "boy", "boys", "men", "gents", "only boys"],
        "mixed": ["coed", "mixed", "both genders", "any gender", "shared with guys and girls"]
    }
    
    # --- Compile phrase matchers ---
    req_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for label, phrases in requirement_phrases.items():
        req_matcher.add(label, [nlp(text) for text in phrases])
    
    gender_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for label, phrases in gender_phrases.items():
        gender_matcher.add(label, [nlp(text) for text in phrases])
    
    # --- Helper functions ---
    def get_requirement(text):
        doc = nlp(text)
        matches = req_matcher(doc)
        if matches:
            return sorted(set([nlp.vocab.strings[m[0]] for m in matches]))[0]
        return ""
    
    def get_gender(text):
        doc = nlp(text)
        matches = gender_matcher(doc)
        if matches:
            return sorted(set([nlp.vocab.strings[m[0]] for m in matches]))[0]
        return ""
    
    def extract_move_in(text):
        m = re.search(r"(move[-\s]?in\s*(date)?[:\-]?\s*(from|on)?\s*[a-zA-Z]{3,9}\s*\d{0,2}|\bnov\b|\bdec\b|\bjan\b)", text.lower())
        return m.group(0) if m else ""
    
    def extract_rent(text):
        m = re.search(r"(\$|usd)\s?\d{3,5}", text.lower())
        return m.group(0) if m else ""
    
    def extract_contacts(text):
        nums = re.findall(r"\+?\d[\d\s\-()]{8,}", text)
        clean = [re.sub(r"\D", "", n) for n in nums]
        return ", ".join(sorted(set(clean)))
    
    def extract_area(text):
        areas = ["fenway", "mission hill", "roxbury", "brookline", "longwood",
                 "allston", "cambridge", "somerville", "brighton", "back bay"]
        for a in areas:
            if a in text.lower():
                return a.title()
        return ""
    
    def extract_accom_type(text):
        text_l = text.lower()
        m = re.search(r"(\d+)\s*(bhk|bed|bedroom|br)", text_l)
        if m:
            return m.group(0)
        for typ in ["studio", "shared", "private", "entire", "master", "single"]:
            if typ in text_l:
                return typ
        return ""
    
    def extract_utilities(text):
        t = text.lower()
        return {
            "laundry": "yes" if "laundry" in t else "",
            "heating": "yes" if "heat" in t or "heating" in t else "",
            "water": "yes" if "water" in t else "",
            "utilities_text": ", ".join(
                u for u in ["wifi", "gas", "electricity", "hot water", "ac", "internet", "microwave", "oven", "washer"]
                if u in t
            )
        }
    
    # --- Parse WhatsApp chat ---
    # print(" Parsing WhatsApp chat file...")
    msg_start = re.compile(
        r"^\[?(\d{1,2}/\d{1,2}/\d{2,4}),?\s*([0-9: ]+[APMapm]*)\]?\s*[-–]?\s*(.*?):\s*(.*)"
    )
    messages, current = [], None
    
    with open(input_file, encoding="utf-8", errors="ignore") as f:
        lines = [line.replace('\r', '').strip() for line in f if line.strip()]
    
    for line in lines:
        m = msg_start.match(line)
        if m:
            if current:
                messages.append(current)
            date, time, sender, text = m.groups()
            current = {"timestamp": f"{date} {time}", "sender": sender.strip(), "message": text.strip()}
        else:
            if current:
                current["message"] += " " + line.strip()
    if current:
        messages.append(current)
    
    # print(f"Parsed {len(messages)} total messages from chat file.")
    
    # --- Semantic-aware filtering ---
    positive_keywords = [
        "accommodation", "room", "bhk", "sublet", "lease", "move-in", "move in", "tenant",
        "rent", "spot", "apartment", "housing", "flat", "available from", "roommate"
    ]
    
    negative_keywords = [
        "for sale", "selling", "pickup", "dm for", "lamp", "chair", "table", "bike",
        "bed frame", "microwave", "sofa", "mattress", "furniture", "sale", "discount",
        "giveaway", "chopper", "move out"
    ]
    
    # Semantic seeds
    # print("Filtering housing-related messages...")
    housing_seed = nlp("Looking for room, accommodation, or apartment for rent or lease.")
    sale_seed = nlp("Selling furniture, appliances, or personal items for pickup or sale.")
    
    filtered = []
    for m in tqdm(messages, desc="Filtering relevant posts"):
        msg_lower = m["message"].lower()
        doc = nlp(m["message"])
        
        # keyword checks
        has_pos = any(p in msg_lower for p in positive_keywords)
        has_neg = any(n in msg_lower for n in negative_keywords)
        
        # semantic similarity checks
        sim_housing = doc.similarity(housing_seed)
        sim_sale = doc.similarity(sale_seed)
        
        # logic: keep if it's housing-related
        if has_pos or sim_housing > sim_sale:
            filtered.append(m)
        elif has_neg and sim_sale > sim_housing:
            continue  # skip obvious sale posts
    
    # print(f"Filtered {len(filtered)} housing-related messages out of {len(messages)} total.")
    
    # Save filtered posts before NLP extraction
    with open(filtered_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "sender", "message"])
        writer.writeheader()
        writer.writerows(filtered)
    # print(f"Saved semantically filtered listings to {filtered_file}")
    
    # --- Extract structured data ---
    # print("Extracting structured fields...")
    structured = []
    for m in tqdm(filtered, desc="Extracting structured fields"):
        text = m["message"]
        utils = extract_utilities(text)
        structured.append({
            "timestamp": m["timestamp"],
            "sender": m["sender"],
            "requirement": get_requirement(text),
            "accom_type": extract_accom_type(text),
            "gender": get_gender(text),
            "food_pref": "veg" if "veg" in text.lower() and "non" not in text.lower() else ("non-veg" if "non" in text.lower() else ""),
            "furnished": "furnished" if "furnished" in text.lower() else ("semi-furnished" if "semi" in text.lower() else ""),
            "red_eye": "yes" if "red eye" in text.lower() or "redeye" in text.lower() else "",
            "area": extract_area(text),
            "move_in_date": extract_move_in(text),
            "rent_amount": extract_rent(text),
            "contact_numbers": extract_contacts(text),
            "laundry": utils["laundry"],
            "heating": utils["heating"],
            "water": utils["water"],
            "utilities_text": utils["utilities_text"],
            "message": text
        })
    
    # --- Save structured data ---
    fields = [
        "timestamp","sender","requirement","accom_type","gender","food_pref","furnished",
        "red_eye","area","move_in_date","rent_amount","contact_numbers",
        "laundry","heating","water","utilities_text","message"
    ]
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(structured)
    
    # print(f"Saved {len(structured)} structured listings to {output_file}")


# Example usage:
# extract_housing_listings("/content/_chat 2.txt")
# 
# This will create two CSV files:
# - relevant_listings.csv (filtered messages)
# - structured_listings_nlp.csv (structured data)