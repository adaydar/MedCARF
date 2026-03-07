#Generate coarse report from phrases

import json
import random
from collections import Counter

LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices"
]

def load_ground_truth_reports(json_path):
    """
    Load all textual reports from the JSON dataset file.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract reports from "train" split
    reports = [item["report"] for item in data["train"]]

    return reports

ground_truth_reports = load_ground_truth_reports("../classification/chest/iu_xray/annotation_labels.json")
    
def load_phrase_dict(path):
    with open(path, "r") as f:
        return json.load(f)

phrase_dict = load_phrase_dict("common_files/label_phrases.json")

def build_phrase_frequency(ground_truth_reports, phrase_dict):

    freq = {label: Counter() for label in phrase_dict.keys()}

    for report in ground_truth_reports:
        for label, phrases in phrase_dict.items():
            for p in phrases:
                if p in report:
                    freq[label][p] += 1

    return freq

phrase_freq = build_phrase_frequency(ground_truth_reports, phrase_dict)

def most_frequent_phrase(label, phrase_dict, phrase_freq):
    phrases = phrase_dict.get(label, [])
    if not phrases:
        return None

    freq = phrase_freq.get(label, {})

    if not freq:
        return phrases[0]   # fallback

    # Choose most frequent among available phrases
    return max(freq, key=freq.get)
    
def generate_report_batch(pred_labels_batch, phrase_dict, phrase_freq):

    reports = []

    for pred_labels in pred_labels_batch:
        sentences = []

        for i, val in enumerate(pred_labels):
            if val == 1:
                label_name = LABELS[i]

                chosen = most_frequent_phrase(label_name, phrase_dict, phrase_freq)

                if chosen:
                    sentences.append(chosen)
                else:
                    sentences.append(f"{label_name} present.")

        if not sentences:
            reports.append("No significant abnormalities detected.")
        else:
            reports.append(" ".join(sentences))

    return reports
