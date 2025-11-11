"""
A self-contained script for running the HealthBenchCompletenessGrader.

This script combines the necessary classes from the repository into a single file
to grade the completeness of medical answers based on dynamically generated criteria.
"""
import os
import json
import asyncio
import logging
from typing import List, Any, Dict, Optional
from abc import ABC, abstractmethod
import argparse
import pickle
from pathlib import Path

import torch
import jsonlines
import numpy as np
from openai.types.chat.chat_completion import ChatCompletion
from sentence_transformers import SentenceTransformer, util

from utilities import APIModel

# --- Setup Logging ---
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("StandaloneGrader")
logging.getLogger("httpx").setLevel(logging.WARNING)  # Silence noisy httpx logs
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

os.environ["TOKENIZERS_PARALLELISM"] = "false" # Silence tokenizer warnings


# --- Similarity and Caching Helpers ---
def get_embeddings(dataset: List[Dict], model: SentenceTransformer, cache_path: Optional[str] = None) -> np.ndarray:
    """Computes or loads cached embeddings for a dataset."""
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading embeddings from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            embeddings = pickle.load(f)
        if embeddings.shape[0] != len(dataset):
            raise ValueError("Cached embeddings size does not match dataset size.")
        logger.info(f"Loaded {embeddings.shape[0]} embeddings from cache.")
    else:
        logger.info(f"Generating embeddings for {len(dataset)} items...")
        questions = [item['question'] for item in dataset]
        # Encode with specified batch size and progress bar
        embeddings = model.encode(
            questions,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=128
        )
        if cache_path:
            logger.info(f"Saving embeddings to cache: {cache_path}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(embeddings, f)
    return embeddings


def find_most_similar(query_embedding: np.ndarray, corpus_embeddings: np.ndarray, corpus: List[Dict], top_k: int) -> List[Dict]:
    """Finds the top_k most similar items from the corpus to the query using dot product."""
    # Using dot_score for similarity as embeddings are normalized
    scores = util.dot_score(query_embedding, corpus_embeddings)[0]
    top_results = np.argpartition(-scores, range(top_k))[0:top_k]
    top_results_docs = []
    for idx in top_results:
        doc = corpus[idx]
        top_results_docs.append({
            "id": doc["id"],
            "question": doc["question"],
            "criteria": doc.get("criteria", []),
            "similarity_score": float(scores[idx].item())
        })
    return top_results_docs


# --- HealthBench Response Grading Infrastructure ---
class RubricItem:
    """Helper class to structure rubric items for grading."""

    def __init__(self, criterion: str, points: float, tags: list[str]):
        self.criterion = criterion
        self.points = points
        self.tags = tags

    def __str__(self):
        return f"[{self.points}] {self.criterion}"

    def to_dict(self):
        return {"criterion": self.criterion, "points": self.points, "tags": self.tags}


GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Instructions
Return a valid JSON object with the following fields: "explanation" and "criteria_met". Ensure all keys and string values are enclosed in double quotes.
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>
""".strip()


class HealthBenchResponseGrader(APIModel):
    """Grades responses against a provided rubric."""
    grader_name = "response-grader-healthbench"

    async def __call__(
            self,
            inputs: List[Dict[str, Any]],
            **kwargs,
    ) -> List[Dict[str, Any]]:
        grouped_output = {}
        augmented_inputs = []

        for item in inputs:
            for metric_item in self.get_criteria(item["id"]):
                rubric_item = RubricItem(**metric_item)
                new_item = item.copy()
                new_item["rubric_item"] = rubric_item
                augmented_inputs.append(new_item)

                if item["id"] not in grouped_output:
                    grouped_output[item["id"]] = {
                        "id": item["id"],
                        self.grader_name: {
                            "total_points": 0,
                            "score": 0,
                            "scoring_model": self.model_name,
                            "meta": [],
                            "icl_examples": item.get("icl_examples", []),
                        }
                    }
                grouped_output[item["id"]][self.grader_name]["total_points"] += rubric_item.points

        raw_output = await super().__call__(augmented_inputs, **kwargs)

        for output in raw_output:
            if output is None: continue  # Skip failed API calls
            details = output["rubric_item"].to_dict()
            details.update(output["rubric_grade"])
            grouped_output[output["id"]][self.grader_name]["meta"].append(details)
            if details.get("criteria_met", False):
                grouped_output[output["id"]][self.grader_name]["score"] += details["points"]

        return list(grouped_output.values())

    async def _individual_call(self, item: Dict[str, Any]) -> ChatCompletion:
        rubric_item = item["rubric_item"]
        conversation = f"""user: {item["question"]}\n\nassistant: {item["response"]}"""
        formatted_input = GRADER_TEMPLATE.replace("<<conversation>>", conversation)
        formatted_input = formatted_input.replace("<<rubric_item>>", str(rubric_item))
        messages = [{"role": "user", "content": formatted_input}]

        completion = await self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            response_format={"type": "json_object"},
            **self.default_params
        )
        return completion

    def _process_completions(
            self,
            batch: List[Dict[str, Any]],
            completions: List[ChatCompletion]
    ) -> List[Dict[str, Any]]:
        processed = []
        for item, completion in zip(batch, completions):
            if completion is None: continue
            content = completion.choices[0].message.content
            try:
                rubric_grade = json.loads(content)
            except json.decoder.JSONDecodeError as err:
                logger.warning(f"Could not decode response: {completion}. {err}")
                rubric_grade = {"explanation": f"Error decoding '{content}'"}
            item["rubric_grade"] = rubric_grade
            processed.append(item)
        return processed

    @abstractmethod
    def get_criteria(self, item_id: Optional[str]) -> List[Dict[str, Any]]:
        pass


# --- HealthBench Completeness-Specific Classes ---

class HealthBenchCompletenessCriteria(APIModel):
    """Generates a list of criteria needed for an answer to be complete."""

    def __init__(self, **kwargs):
        self.prompt_template_base = """
# Instructions
Your task is to define a rubric evaluating the **completeness** of an answer to a medical question. We define **completeness** as:
whether a response includes all important information needed to be safe and helpful to the user. Even if accurate, a response that is incomplete (e.g., omitting key steps or red flags) can still result in low-quality advice or harm.
You will be provided a medical question from a user. Write criteria that the assistant's next message will need to include for the message to be **complete**.
Return your answer in JSONLines format. Do not include any other text in the response.
```json
[
    {
        "criteria": ""
    },
    {
        "criteria": ""
    }
]
```
# Examples
{examples}
# Your Task
{task}
"""
        self.task_template = "## Question\n{question}\n\n## Criteria\n{criteria}".strip()
        super().__init__(**kwargs)

    def _format_icl_prompt(self, examples: List[Dict[str, Any]]) -> str:
        """Formats the base prompt template with provided examples."""
        if not examples:
            return self.prompt_template_base.replace("# Examples\n{examples}\n", "")

        formatted_examples = []
        for idx, example in enumerate(examples):
            ex = self._format_task(example, is_example=True)
            ex = ex.replace("##", "###")
            ex = f"## Example {idx + 1}:\n{ex}"
            formatted_examples.append(ex)
        return self.prompt_template_base.replace("{examples}", "\n\n".join(formatted_examples))

    async def _individual_call(self, item: Dict[str, Any]):
        # Dynamically format the prompt with the most similar examples for this specific item
        icl_examples = item.get("icl_examples", [])
        prompt_template = self._format_icl_prompt(icl_examples)

        formatted_task = self._format_task(item)
        formatted_input = prompt_template.replace("{task}", formatted_task)
        messages = [{"role": "user", "content": formatted_input}]

        completion = await self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            **self.default_params
        )
        return completion

    def _process_completions(self, batch: List[Dict[str, Any]], completions: List[ChatCompletion]) -> List[
        Dict[str, Any]]:
        output = []
        for b, completion in zip(batch, completions):
            if completion is None: continue
            content = completion.choices[0].message.content
            # Clean the content before loading
            criteria_text = content.strip().replace("```json", "").replace("```", "").strip()
            try:
                criteria_list = json.loads(criteria_text)
                o = {"id": b["id"], "criteria": [c["criteria"] for c in criteria_list]}
            except (json.decoder.JSONDecodeError, TypeError):
                logger.warning(f"Could not decode criteria JSON for ID '{b['id']}': {criteria_text}")
                o = {"id": b["id"], "criteria": []}
            o["icl_examples"] = b.get("icl_examples", [])
            output.append(o)
        return output

    def _format_task(self, task: Dict[str, Any], is_example: bool = False) -> str:
        criteria = ""
        if is_example:
            criteria = [{"criteria": c} for c in task.get("criteria", ["None"])]
            criteria = json.dumps(criteria)
        return self.task_template.format(question=task["question"], criteria=criteria)


class HealthBenchCompletenessGrader(HealthBenchResponseGrader):
    """
    Main class that first generates completeness criteria and then grades a response against them.
    """
    grader_name = "completeness-healthbench"

    def __init__(self, **kwargs):
        self.criteria_generator = HealthBenchCompletenessCriteria(**kwargs)
        self.id_to_criteria = None
        super().__init__(**kwargs)

    async def __call__(self, inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        logger.info("Step 1: Generating completeness criteria using dynamic examples...")
        generated_criteria = await self.criteria_generator(inputs)
        self.id_to_criteria = {c["id"]: c["criteria"] for c in generated_criteria}

        logger.info("Step 2: Grading responses based on generated criteria...")
        output = await super().__call__(inputs, **kwargs)
        return output

    def get_criteria(self, item_id: Optional[str]) -> List[Dict[str, Any]]:
        """Provides the generated criteria for a given item ID."""
        item_criteria_list = self.id_to_criteria.get(item_id, [])
        return [{"criterion": c, "points": 1, "tags": ["completeness"]} for c in item_criteria_list]


##########
# Main
##########
def parse_args():
    parser = argparse.ArgumentParser(description="Grade medical chatbot responses for completeness using dynamic ICL.")
    parser.add_argument("--input_file", type=str, required=True, help="Input file in JSONLines format to be graded.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory.")
    parser.add_argument("--icl_dataset_file", type=str, required=True, help="JSONLines file with examples for dynamic ICL.")
    parser.add_argument("--num_examples", type=int, default=2, help="Number of similar ICL examples to use.")
    parser.add_argument("--similarity_model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model for similarity.")
    parser.add_argument("--cache_embeddings", action="store_true", help="Cache ICL dataset embeddings to a file.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output.")
    parser.add_argument("--server_path", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"))
    return parser.parse_args()


async def main():
    """
    Main asynchronous function to run the grading process.
    """
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    logger.debug(f"Running with arguments: {args}")

    # Save the args to output directory for reference
    os.makedirs(args.output_dir, exist_ok=True)
    with open(Path(args.output_dir) / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    try:
        # 1. Load ICL and test data
        with jsonlines.open(args.icl_dataset_file) as reader:
            icl_dataset = list(reader)
        with jsonlines.open(args.input_file) as reader:
            test_data = list(reader)

        # 2. Setup similarity model and get embeddings for the ICL dataset
        # Use GPU for embedding if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device} for sentence embeddings.")
        similarity_model = SentenceTransformer(args.similarity_model, device=device)

        cache_file = None
        if args.cache_embeddings:
            model_name_slug = args.similarity_model.replace("/", "_")
            cache_file = Path(args.output_dir) / f"embeddings_{model_name_slug}.pkl"

        icl_embeddings = get_embeddings(icl_dataset, similarity_model, cache_path=cache_file)

        # 3. Augment test data with dynamically selected ICL examples
        logger.info(f"Finding the top {args.num_examples} most similar examples for each test question...")
        test_questions = [item['question'] for item in test_data]
        test_embeddings = similarity_model.encode(
            test_questions,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=128
        )

        augmented_test_data = []
        for i, item in enumerate(test_data):
            similar_examples = find_most_similar(test_embeddings[i], icl_embeddings, icl_dataset, top_k=args.num_examples)
            new_item = item.copy()
            new_item['icl_examples'] = similar_examples
            augmented_test_data.append(new_item)

        # 4. Initialize and run the grader.
        grader_args = {
            "model_name": args.model_name,
            "model_server": args.server_path,
            "api_key": args.api_key
        }
        completeness_grader = HealthBenchCompletenessGrader(**grader_args)
        graded_responses = await completeness_grader(augmented_test_data)

        # 5. Print and save the results.
        # Only print a demo of 2 results to avoid flooding the console
        logger.info("\n--- Grading Results ---")
        for response in graded_responses[:2]:
            grade_info = response['completeness-healthbench']
            print(f"\nID: {response['id']}")
            print(f"Score: {grade_info['score']} out of {grade_info['total_points']} points")
            print("Generated Criteria & Grades:")
            for i, criteria_item in enumerate(grade_info['meta']):
                print(f"  - Criterion {i + 1}: {criteria_item['criterion']}")
                print(f"    - Met?: {criteria_item['criteria_met']}")
                print(f"    - Explanation: {criteria_item['explanation']}")

        output_file = Path(args.output_dir) / "omissions_healthbench-icl_output.jsonl"
        with jsonlines.open(output_file, "w") as writer:
            writer.write_all(graded_responses)
        logger.info(f"\nFull results saved to {output_file}")

    except (ValueError, FileNotFoundError) as e:
        logger.error(e)
    except Exception as e:
        logger.error(f"An unexpected error occurred during grading: {e}")


# --- Runnable Example ---
if __name__ == "__main__":
    # Use a single asyncio.run() to manage the entire async process
    asyncio.run(main())

