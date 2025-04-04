import pandas as pd
import random
import csv
from datetime import datetime, timedelta

# --- Configuration ---
NUM_RECORDS = 1000
OUTPUT_FILE = "dummy_data.csv"

# --- Data Generation Options ---
SEX_OPTIONS = ["Male", "Female"]
PROCEDURE_OPTIONS = [
    "Appendectomy", "Cholecystectomy", "Knee Replacement", "Hip Replacement",
    "Hernia Repair", "Wound Debridement", "CABG", "Skin Graft",
    "Abscess Drainage", "Exploratory Laparotomy", "Cesarean Section",
    "Fracture Repair", "Biopsy", "Central Line Placement"
]
WOUND_TYPES = [
    "Surgical Incision", "Traumatic Laceration", "Pressure Ulcer Stage 2",
    "Diabetic Foot Ulcer", "Burn Wound", "Abrasion", "Puncture Wound",
    "Venous Stasis Ulcer"
]
WOUND_APPEARANCE_GOOD = [
    "clean, dry, intact", "edges well approximated", "healing by primary intention",
    "granulating tissue present", "no drainage noted", "sutures/staples intact",
    "minimal serous exudate"
]
WOUND_APPEARANCE_BAD = [
    "moderate erythema surrounding", "significant edema noted", "warmth to touch",
    "purulent drainage observed", "malodorous", "dehiscence present",
    "necrotic tissue/slough noted", "copious serosanguinous drainage",
    "undermining present", "increased tenderness reported"
]
DRUGS_ANTIBIOTICS = [
    "Vancomycin", "Piperacillin/Tazobactam", "Cefazolin", "Clindamycin",
    "Metronidazole", "Levofloxacin", "Doxycycline", "None prescribed",
    "Keflex PO", "Augmentin PO"
]
DRUGS_ANALGESICS = [
    "Acetaminophen", "Ibuprofen", "Oxycodone", "Morphine PCA", "Hydromorphone",
    "Tramadol", "Gabapentin", "Ketorolac IV"
]
OTHER_NOTES = [
    "Patient afebrile.", "Patient febrile Tmax 38.6C.", "WBC count stable.",
    "WBC elevated at 15.5.", "Blood cultures pending.", "Blood cultures negative.",
    "Pain controlled.", "Reports increasing pain.", "Mobilizing well.",
    "Diet advanced.", "NPO.", "Dressing changed.", "Vitals stable.",
    "Complains of nausea."
]

# --- Helper Function to Generate Notes ---
def generate_clinical_note(procedure, wound_type):
    notes = []
    post_op_day = random.randint(1, 14)
    notes.append(f"Post-operative day {post_op_day} following {procedure}.")

    # Wound description
    has_wound_details = random.random() < 0.9 # 90% chance of wound details
    if has_wound_details:
        notes.append(f"Assessment of {wound_type}.")
        # Decide if signs of infection are present (higher chance for some procedures/wound types)
        infection_chance = 0.15
        if procedure in ["Abscess Drainage", "Wound Debridement"] or "Ulcer" in wound_type:
            infection_chance = 0.4
        if "purulent" in random.choice(WOUND_APPEARANCE_BAD): # Ensure some definite infections
             infection_chance = 0.95

        if random.random() < infection_chance: # Signs of potential infection
            num_bad_signs = random.randint(1, 3)
            notes.append(f"Wound appearance: {', '.join(random.sample(WOUND_APPEARANCE_BAD, num_bad_signs))}.")
            if random.random() < 0.7: # Higher chance of antibiotics if infection signs
                 notes.append(f"Antibiotics: Currently on {random.choice([d for d in DRUGS_ANTIBIOTICS if d != 'None prescribed'])}.")
            else:
                 notes.append(f"Antibiotics: {random.choice(DRUGS_ANTIBIOTICS)}.")

        else: # No clear signs of infection
            num_good_signs = random.randint(1, 2)
            notes.append(f"Wound appearance: {', '.join(random.sample(WOUND_APPEARANCE_GOOD, num_good_signs))}.")
            if random.random() < 0.85: # Lower chance of antibiotics if no infection signs
                 notes.append(f"Antibiotics: {random.choice([d for d in DRUGS_ANTIBIOTICS if d == 'None prescribed'])}.")
            else:
                 notes.append(f"Antibiotics: {random.choice(DRUGS_ANTIBIOTICS)}.")

    # Add other details
    notes.append(f"Pain management: {random.choice(DRUGS_ANALGESICS)}.")
    num_other_notes = random.randint(1, 3)
    notes.extend(random.sample(OTHER_NOTES, num_other_notes))

    return " ".join(notes)

# --- Generate Data ---
data = []
start_date = datetime.now() - timedelta(days=180) # Start date for admission

print(f"Generating {NUM_RECORDS} synthetic records...")

for i in range(NUM_RECORDS):
    patient_id = 10000 + i
    age = random.randint(18, 90)
    sex = random.choice(SEX_OPTIONS)
    bmi = round(random.uniform(18.0, 45.0), 1)
    procedure = random.choice(PROCEDURE_OPTIONS)

    # Assign a somewhat relevant wound type
    if "Wound" in procedure or "Abscess" in procedure or "Graft" in procedure:
        wound_type = random.choice([wt for wt in WOUND_TYPES if "Surgical" not in wt or "Ulcer" in wt or "Burn" in wt])
    elif procedure in ["Appendectomy", "Cholecystectomy", "Hernia Repair", "CABG", "Exploratory Laparotomy", "Cesarean Section"]:
        wound_type = "Surgical Incision"
    elif "Replacement" in procedure or "Fracture" in procedure:
         wound_type = random.choice(["Surgical Incision", "Puncture Wound"]) # e.g., external fixator pins
    else:
        wound_type = random.choice(WOUND_TYPES) # Default random

    notes = generate_clinical_note(procedure, wound_type)

    # Add admission date (optional, but adds context)
    admission_date = start_date + timedelta(days=random.randint(0, 170))
    admission_date_str = admission_date.strftime('%Y-%m-%d')

    data.append({
        "patient_id": patient_id,
        "admission_date": admission_date_str,
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "procedure_type": procedure,
        "clinical_notes": notes
    })

# --- Create DataFrame and Save CSV ---
df = pd.DataFrame(data)

# Optional: Shuffle rows
df = df.sample(frac=1).reset_index(drop=True)

try:
    df.to_csv(OUTPUT_FILE, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Successfully generated and saved {NUM_RECORDS} records to {OUTPUT_FILE}")
    print("\n--- Sample Data ---")
    print(df.head())
    print("\n--- Notes Sample ---")
    print(df['clinical_notes'].iloc[0])

except Exception as e:
    print(f"Error saving file: {e}")

