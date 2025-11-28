"""
Convert the annotation JSON file to JSONL format

Author: Mahsa Yarmohammadi
Modified: Alexandra DeLucia
Modified: Sonal Joshi

JSON schema:

* `completions` (**array**): A list of objects, where each object represents an annotation or completion of the task.
    * `created_username` (**string**): The user who created the annotation.
    * `created_ago` (**string** - Timestamp): The date and time the annotation was created.
    * `result` (**array**): A list containing the specific details of the annotation.
        * `value` (**object**): Contains the actual annotation data.
            * `choices` (**array** of **strings**): The option(s) selected by the annotator (e.g., `["High"]`).
            * `confidence` (**integer**): A numerical value representing the annotator's confidence.
        * `id` (**string**): A unique ID for the result.
        * `from_name` (**string**): The name of the control tag that produced the result.
        * `to_name` (**string**): The name of the object tag that was annotated.
        * `type` (**string**): The type of annotation (e.g., `"choices"`).
        * `sections` (**array**): (Empty in the example, likely for more complex annotations).
    * `pk` (**string**): A primary key for the completion, likely from a database.
    * `honeypot` (**boolean**): A flag, often used to detect automated bots.
    * `lead_time` (**integer**): The time it took for the user to submit the annotation, likely in seconds.
    * `id` (**integer**): A unique numerical ID for the completion.
    * `confidence_range` (**array** of **integers**): The minimum and maximum possible values for the confidence score.
    * `submitted_at` (**string** - Timestamp): The date and time the annotation was submitted.
    * `updated_at` (**string** - Timestamp): The date and time the annotation was last updated.
    * `updated_by` (**string**): The user who last updated the annotation.
* `predictions` (**array**): A list of prediction objects, likely from a model (empty in the example).
* `created_at` (**string** - Datetime): The creation date and time of the overall data record.
* `created_by` (**string**): The user who created the data record.
* `data` (**object**): The source data that was being annotated.
    * `question` (**string**): The question or prompt that the response addresses.
    * `response` (**string**): The detailed text response that is the subject of the annotation.
    * `title` (**string**): A title or identifier for the data record.
* `id` (**integer**): The unique identifier for the overall data record.
* `sections` (**array**): (Empty in the example).
"""
import json
import csv
import os
import ast
import re
from typing import List, Dict, Any, Tuple, Union, Optional
import logging
import argparse
import unicodedata
import hashlib
import base64
from io import StringIO

from tqdm import tqdm
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


##############
# Helpers
##############
def parse_sentences(passage: str) -> List[Dict[str, Any]]:
    """Parse the sentences from a passage"""
    doc = nlp(passage)
    sentences = []
    # sent is a spacy span object https://spacy.io/api/span#init
    # span start/end is based on token index [sent.start, sent.end)
    # convert token index to character to match the annotations
    for sent in doc.sents:
        sentences.append({
            "text": sent.text,
            "span_start": sent.start_char,
            "span_end": sent.end_char
        })
    return sentences


def character_overlap(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
    """
    Determine if two spans overlap using standard interval logic.
    Assumes spans are [start, end) where end is exclusive.
    """
    # They overlap if the start of each span is before the end of the other.
    return span1[0] < span2[1] and span2[0] < span1[1]


def trim_span(passage: str, span: Tuple[int, int]) -> Tuple[int, int]:
    """
    Adjusts a span's start and end characters to exclude leading/trailing whitespace.
    """
    start, end = span
    span_text = passage[start:end]

    # Calculate how much leading whitespace to remove
    leading_spaces = len(span_text) - len(span_text.lstrip())
    new_start = start + leading_spaces

    # Calculate how much trailing whitespace to remove
    trailing_spaces = len(span_text) - len(span_text.rstrip())
    new_end = end - trailing_spaces
    logger.debug(f"Trimming whitespace from <<{span_text=}>> ({start=}, {end=}) to ({new_start=}, {new_end=})")
    return new_start, new_end


def flatten_list(items: Union[None, List[str]]) -> Union[None, str]:
    flattened = []
    for item in items:
        # if pd.isna(item):
        if not item:
            continue
        flattened.extend(item)
    flattened = list(set([i.strip() for i in flattened]))
    if len(flattened) > 1:
        logger.debug("More than one item found. Combining: {}".format(flattened))
    if flattened:
        return ". ".join(flattened)
    else:
        return None


def extract_certainty(row: pd.Series) -> Dict[str, Any]:
    certainty_annotation = {
        "annotation_id": row.annotation_id,
        "model_certainty": None,
    }
    if row.from_name == "llmcertainty_opts":
        certainty_annotation["model_certainty"] = row.value["choices"][0].strip()
    return certainty_annotation


def extract_omissions(row: pd.Series) -> Dict[str, Any]:
    omissions_annotation = {
        "annotation_id": row.annotation_id,
        "omission_severity": None,
        "omission_description": None,
    }
    if row.from_name == "completeness_opts":
        omissions_annotation["omission_severity"] = row.value["choices"][0].strip()
    if row.from_name == "answer_omission":
        omissions_annotation["omission_description"] = [o.strip() for o in row.value["text"]]
    return omissions_annotation


def extract_factuality(row: pd.Series) -> Dict[str, Any]:
    factuality_annotation = {
        "annotation_id": row.annotation_id,
        "factuality_severity": None,
        "factuality_text": None,
        "factuality_span": None,
        "factuality_explanation": None
    }
    if row.from_name == "factuality_label":
        factuality_annotation["factuality_severity"] = row.value["labels"][0]
        factuality_annotation["factuality_text"] = row.value["text"]
        factuality_annotation["factuality_span"] = (row.value["start"], row.value["end"])
    if row.from_name == "answer_factuality":
        factuality_annotation["factuality_span"] = (row.value["start"], row.value["end"])
        factuality_annotation["factuality_explanation"] = row.value["text"]
    return factuality_annotation


def extract_reference(row: pd.Series) -> Dict[str, Any]:
    reference_annotation = {
        "annotation_id": row.annotation_id,
        "reference_attribution": None,
    }
    if row.from_name == "answer_ref":
        reference_annotation["reference_attribution"] = row.value["text"]
    return reference_annotation


def extract_comments(row: pd.Series) -> Dict[str, Any]:
    comments_annotation = {
        "annotation_id": row.annotation_id,
        "comments": None,
    }
    if row.from_name == "general_comment":
        comments_annotation["comments"] = row.value["text"]
    return comments_annotation


def parse_settings_from_title(row: pd.Series) -> Dict[str, Any]:
    # Parse the domain, model, and question ID from the title
    # Ex. MH-gemma2-q50.1.h3-iaa, MH-olmo2-q10
    title = row.title.replace("-iaa", "")
    domain, model, question_settings = title.split("-")
    if "." in question_settings:
        base_question, context_id, hypothesis = question_settings.split(".")
    else:
        base_question = question_settings
        context_id = None
        hypothesis = None
    return {
        "annotation_id": row.annotation_id,
        "domain": domain,
        "model": model,
        "base_question_id": base_question,
        "context_id": context_id,
        "hypothesis_id": hypothesis,
    }


def extract_all_annotations(row: pd.Series) -> pd.Series:
    """Combines all extraction logic into a single function."""
    certainty = extract_certainty(row)
    omissions = extract_omissions(row)
    factuality = extract_factuality(row)
    reference = extract_reference(row)
    comments = extract_comments(row)
    title = parse_settings_from_title(row)

    # Combine all dictionaries, ensuring annotation_id is kept only once
    combined = {
        **title,
        **certainty,
        **omissions,
        **factuality,
        **reference,
        **comments,
    }
    return pd.Series(combined)


def create_unique_id(str_to_hash: str, length: int = 16) -> str:
    """Creates a URL-safe Base64 ID."""
    # Get the raw bytes of the hash, not the hex digest
    hash_bytes = hashlib.sha256(str_to_hash.encode('utf-8')).digest()

    # Encode the bytes in Base64 and decode to a string
    base64_id = base64.urlsafe_b64encode(hash_bytes).decode('utf-8')
    return base64_id


def convert_json_to_jsonl(input_file: str) -> None:
    """Convert the raw annotation JSON file to a processed JSONL file."""
    # Load raw annotations from JSON
    try:
        # 1. Open the file and read the raw text with the correct initial encoding
        with open(input_file, 'r', encoding="utf-8") as f:
            raw_text = f.read()
            # NFKD normalization breaks down characters into their base components.
            # The encode/decode step removes any remaining non-ASCII characters.
            raw_text = unicodedata.normalize('NFKD', raw_text).encode('ascii', 'ignore').decode('utf-8')

        # 2. Load the fully cleaned string into pandas using StringIO.
        raw_data_df = pd.read_json(StringIO(raw_text))
    except (FileNotFoundError, UnicodeDecodeError, ValueError) as err:
        logger.error(f"Issue opening {input_file}: {err}")
        exit(1)
    logger.debug(f"Loaded {len(raw_data_df)} raw annotations from {input_file}")

    # Unroll the annotations into individual rows (in completions.result field)
    annotations_df = []
    for index, row in raw_data_df.iterrows():
        for completion in row.completions:
            if not completion.get("submitted_at"):
                logger.debug(f"{index} by {completion['created_username']} is not submitted, skipping")
                continue
            for annotation in completion.get("result", []):
                # Check if annotation is actuality submitted
                new_row = {
                    "annotator": completion["created_username"],
                    "created": completion["created_ago"],
                    "submitted": completion["submitted_at"],
                    "last_updated": completion.get("updated_at"),
                    "annotation_id": annotation["id"],
                }
                new_row.update(row.data)
                new_row.update(annotation)
                annotations_df.append(new_row)
    annotations_df = pd.DataFrame(annotations_df)

    logger.debug("Expanding all annotation types at once...")
    # Apply the new function to create a DataFrame with all new columns
    new_cols_df = annotations_df.apply(extract_all_annotations, axis=1)
    annotations_df = pd.concat(
        [annotations_df, new_cols_df.drop(columns=['annotation_id'])],
        axis=1
    )

    # Normalize newline characters
    annotations_df['response'] = annotations_df['response'].str.replace('\r', '')

    # Assign a unique ID based on the title, annotator, question
    annotations_df["unique_id"] = annotations_df.apply(lambda x: create_unique_id("".join([x.title, x.annotator, x.question])), axis=1)

    # Merge annotations by title (id), question, and annotator
    # Note: This assumes that the title and annotator are unique for each annotation.
    # "Question" is added to address the case where multiple entries have the same title but different questions.
    # If there are multiple annotations for the same title and annotator, they will be merged.
    merged_annotations = []
    logger.debug(f"Merging annotations based on unique title, question, and annotator")
    for unique_id, df in tqdm(annotations_df.groupby("unique_id"), desc="Merging annotations"):
        df.sort_values(by="last_updated", ascending=True, inplace=True)

        row = {
            "id": unique_id,  # Unique ID for the task
            "domain": df.iloc[0].domain,
            "model": df.iloc[0].model,
            "base_question_id": df.iloc[0].base_question_id,
            "context_id": df.iloc[0].context_id,
            "hypothesis_id": df.iloc[0].hypothesis_id,
            "title": df.iloc[0].title,
            "annotator": df.iloc[0].annotator,
            "question": df.iloc[0].question,
            "response": df.iloc[0].response,
            "factuality": [],
            "omissions": [],
            "comments": [],
            "references": [],
            "meta": {
                "annotator": df.iloc[0].annotator,
                "created": df.created.unique().tolist(),
                "submitted": df.submitted.dropna().unique().tolist(),
                "last_updated": df.last_updated.unique().tolist()[0],
                "annotation_id": df.annotation_id.unique().tolist(),
            }
        }
        # All entries should have a model confidence/certainty label
        unique_certainty = df.model_certainty.dropna().unique()
        if len(unique_certainty) > 1:
            logger.warning(f"Entry {unique_id=} has multiple model certainties: {unique_certainty}. Taking the first one.")
        row["model_certainty"] = unique_certainty[0] if len(unique_certainty)>0 else None

        # Check for omission labels
        if df.omission_severity.any():
            # Unique values for omission severity
            unique_severity = df.omission_severity.dropna().unique()
            if len(unique_severity) > 1:
                logger.warning(f"Entry {unique_id=} has multiple omission severities: {unique_severity}. Taking the first one.")
            # If there are multiple values, take the first one
            row["omission_severity"] = unique_severity[0]

        # Add omission descriptions
        if df.omission_description.any():
            split_omissions = []
            for omission_list in df.omission_description.dropna().tolist():
                for om in omission_list:
                    om = [o.strip() for o in om.split("\n") if o.strip()]
                    split_omissions.extend(om)
            # Drop duplicates
            omissions = list(set(split_omissions))

            # Further cleaning. Remove "- " and "1)" in front of the omission.
            omissions = [re.sub(r"^-+", "", om).strip() for om in omissions]
            omissions = [re.sub(r"^\d+\)", "", om).strip() for om in omissions]
            omissions = [om for om in omissions if om]
            row["omissions"] = omissions

        # Add factuality annotations. Group by the spans.
        fact_df = df.dropna(subset=["factuality_span"]).groupby(["factuality_span"]).agg({
            "factuality_severity": lambda x: x.dropna().unique()[0],
            "factuality_text": lambda x: x.dropna().unique()[0],
            "factuality_explanation": flatten_list,
        })
        fact_df.rename(columns={
            "factuality_severity": "severity",
            "factuality_text": "highlighted_text",
            "factuality_explanation": "comment",
        }, inplace=True)
        fact_df.reset_index()
        fact_df["span_start"] = fact_df.index.map(lambda x: x[0])
        fact_df["span_end"] = fact_df.index.map(lambda x: x[1])

        # Check for duplicate entries.
        # A factuality annotation is a duplicate if
        # 1. its span is a subset of another span AND
        # 2. it has no comment
        drop_spans = []
        for idx, r in fact_df.iterrows():
            if r.comment:
                continue
            subset_of = fact_df[(fact_df.span_start<=r.span_start)&(fact_df.span_end>=r.span_end)]
            if len(subset_of) > 1:  # It will always match itself, check if it matches more than one
                drop_spans.append(idx)
        fact_df.drop(drop_spans, axis="index", inplace=True)
        if not fact_df.empty:
            row["factuality"] = fact_df.to_dict(orient="records")

        # Add references
        if df.reference_attribution.any():
            references = []
            for ref_list in df.reference_attribution.dropna().tolist():
                for ref in ref_list:
                    ref = [r.strip() for r in ref.split("\n") if r.strip()]
                    references.extend(ref)
            references = list(set(references))
            row["references"] = references

        # Add comments
        if df.comments.any():
            comments = []
            for com_list in df.comments.dropna().tolist():
                for com in com_list:
                    com = [c.strip() for c in com.split("\n") if c.strip()]
                    comments.extend(com)
            comments = list(set(comments))
            row["comments"] = comments


        # Moving counts out of meta
        row["n_omissions"] =  len(row["omissions"])
        row["n_factuality_annotated_spans"] =  len(row["factuality"])

        # row["meta"].update({
        #     "n_omissions": len(row["omissions"]),
        #     "n_annotated_spans": len(row["factuality"])
        # })
        merged_annotations.append(row)
    merged_annotations = pd.DataFrame(merged_annotations)
    logger.debug(f"Merged into {len(merged_annotations)} annotations")

    # Map empty factuality and omissions to None
    merged_annotations["factuality"] = merged_annotations["factuality"].apply(lambda x: x if x else None)
    merged_annotations["omissions"] = merged_annotations["omissions"].apply(lambda x: x if x else None)

    # Save
    output_file = input_file.replace(".json", "_processed.jsonl")
    merged_annotations.to_json(output_file, lines=True, orient="records")
    logger.info(f"Saved {len(merged_annotations)} processed annotations to {output_file}")


def create_senticized(input_file: str, overwrite: bool = False) -> None:
    output_file = input_file.replace(".json", "_processed.jsonl")
    if overwrite or not os.path.exists(output_file):
        logger.debug(f"Creating processed annotations file before senticizing: {output_file}")
        convert_json_to_jsonl(input_file)

    # Load annotations
    try:
        annotations_df = pd.read_json(output_file, lines=True)
    except (FileNotFoundError, UnicodeDecodeError, ValueError) as err:
        logger.error(f"Issue opening {output_file}: {err}")
        exit(1)
    logger.debug(f"Loaded {len(annotations_df)} annotations from {output_file}")

    # For each entry
    senticized_annotations = []
    for idx, row in tqdm(annotations_df.iterrows(), desc="Senticizing annotations"):
        # 1. Split response into sentences.
        sentences = parse_sentences(row.response)
        passage = row.response

        # 2. Align highlighted spans with sentences
        highlighted_spans = row.factuality if row.factuality else []
        false_counter = 0
        for sent_idx, sent in enumerate(sentences):
            logger.debug(f"Sent {sent_idx} {(sent['span_start'], sent['span_end'])}: <<{sent['text']}>>")
            sent["sentence_id"] = sent_idx
            sent_annotations = []
            sent_span = (sent["span_start"], sent["span_end"])

            for highlight in highlighted_spans:
                highlight_text = highlight.get("highlighted_text")
                if not highlight_text:
                    continue

                # Find all occurrences of the highlight text in the passage to get correct indices
                for match in re.finditer(re.escape(highlight_text), passage):
                    # This is the correct span according to Python's string processing
                    correct_span = (match.start(), match.end())

                    # Check if this specific occurrence overlaps with the current sentence
                    if character_overlap(correct_span, sent_span):
                        # If it overlaps, add the original highlight info to the sentence
                        if highlight not in sent_annotations:
                            sent_annotations.append(highlight)

            sent["annotations"] = sent_annotations
            if sent_annotations:
                sent["label"] = False
                false_counter += 1
            else:
                sent["label"] = True

        # Sanity check: there should be at least 1 False sentence per highlighted span
        if false_counter < len(highlighted_spans):
            logger.debug(f"{row.title=} {false_counter=} {len(highlighted_spans)=}")

        new_row = row.copy()
        new_row["meta"]["raw_factuality_spans"] = row["factuality"]
        new_row["factuality"] = sentences
        senticized_annotations.append(new_row)

    senticized_annotations = pd.DataFrame(senticized_annotations)

    # # Remove unecessary columns
    # senticized_annotations = senticized_annotations.drop(columns=["annotation_id","last_updated"])

    output_file = input_file.replace(".json", "_processed_senticized.jsonl")
    senticized_annotations.to_json(output_file, lines=True, orient="records")
    logger.info(f"Saved senticized annotations to {output_file}")


##########
# Main
##########
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Input file in JSON format")
    parser.add_argument("--senticized", action="store_true", help="Create senticized annotations")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing intermediate _processed.jsonl file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    logger.debug(f"Running with arguments: {args}")

    if not args.input_file.endswith(".json"):
        logger.error(f"Input file must be a JSON file and end with .json")
        sys.exit(1)

    logger.info(f"Processing {args.input_file}")
    if args.senticized:
        create_senticized(args.input_file, overwrite=args.overwrite)
    else:
        convert_json_to_jsonl(args.input_file)

    logger.info(f"Finished processing {args.input_file}")