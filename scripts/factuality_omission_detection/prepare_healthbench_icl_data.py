"""
prepare_healthbench_icl_data.py

Download and preprocess the HealthBench evaluation JSONL
from OpenAI's simple-evals and convert it to a compact ICL-ready JSONL.
"""
import os
import logging
import random
import urllib.request
import urllib.parse
import jsonlines
import typing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalQADataset(typing.Iterable):
    def __init__(self, dataset_path: str, n_examples: int = -1, random_state: int = 42):
        self.dataset_path = dataset_path
        self.random_state = random_state
        self.n_examples = n_examples
        self.dataset = self.load_data()

    def __iter__(self) -> typing.Dict[str, typing.Any]:
        for item in self.dataset:
            yield item

    def _load_data(self) -> typing.List[typing.Dict[str, typing.Any]]:
        raise NotImplementedError

    def load_data(self) -> typing.List[typing.Dict[str, typing.Any]]:
        dataset = self._load_data()
        if self.n_examples != -1:
            n_examples = min(self.n_examples, len(dataset))
            random.seed(self.random_state)
            dataset = random.choices(dataset, k=n_examples)
        return dataset

    def get_name(self) -> str:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: typing.Union[int, slice]):
        if isinstance(idx, slice):
            output = [self.dataset[ii] for ii in range(*idx.indices(len(self)))]
            for o in output:
                o["dataset"] = self.get_name()
            return output
        if idx >= len(self):
            raise IndexError
        item = self.dataset[idx]
        item["dataset"] = self.get_name()
        return item


class HealthBenchDataset(MedicalQADataset):
    def __init__(self, dataset_path: str, include_multiturn: bool = False, **kwargs):
        self.include_multiturn = include_multiturn
        super().__init__(dataset_path, **kwargs)

    def _load_data(self) -> typing.List[typing.Dict[str, typing.Any]]:
        dataset = []
        with jsonlines.open(self.dataset_path) as reader:
            for item in reader:
                theme, subtheme = self._parse_theme(item.get("example_tags", []))
                criteria, criteria_meta = self._parse_criteria(item.get("rubrics", []))
                n_turns = len(item.get("prompt", []))
                is_multiturn = n_turns > 1

                if not self.include_multiturn and is_multiturn:
                    continue

                if not criteria:
                    continue

                answer = None
                if item.get("ideal_completions_data"):
                    answer = item["ideal_completions_data"].get("ideal_completion")
                    if answer:
                        answer = answer.strip()

                new_item = {
                    "id": item.get("prompt_id"),
                    "question": self._format_prompt(item.get("prompt", [])),
                    "answer": answer,
                    "criteria": criteria,
                    "omissions": [],
                    "meta": {
                        "n_turns": n_turns,
                        "is_multiturn": is_multiturn,
                        "theme": theme,
                        "subtheme": subtheme,
                        "criteria_meta": criteria_meta,
                    },
                }
                dataset.append(new_item)
        return dataset

    @staticmethod
    def _parse_criteria(rubrics: typing.List[typing.Dict[str, typing.Any]]):
        criteria = []
        criteria_meta = {"points": [], "rubric_idx": []}
        for idx, rubric in enumerate(rubrics):
            tags = rubric.get("tags", [])
            # find axis tag if present
            axis_tag = None
            for t in tags:
                if t.startswith("axis:"):
                    axis_tag = t.replace("axis:", "")
                    break
            if axis_tag != "completeness":
                continue
            if rubric.get("points", 0) < 0:
                continue
            criterion_text = rubric.get("criterion", "")
            if criterion_text.startswith("Judge whether the completion"):
                continue
            criteria.append(criterion_text)
            criteria_meta["points"].append(rubric.get("points", 0))
            criteria_meta["rubric_idx"].append(idx)
        return criteria, criteria_meta

    @staticmethod
    def _format_prompt(prompt: typing.List[typing.Dict[str, str]]) -> str:
        if not prompt:
            return ""
        if len(prompt) == 1:
            return prompt[0].get("content", "").strip()
        combined = ""
        for turn in prompt:
            role = turn.get("role", "")
            content = turn.get("content", "").strip()
            combined += f"{role}: {content}\n"
        return combined.strip()

    @staticmethod
    def _parse_theme(example_tags: typing.List[str]):
        theme, subtheme = None, None
        if not example_tags:
            return theme, subtheme
        if example_tags[0].startswith("theme:"):
            theme = example_tags[0].replace("theme:", "")
        if len(example_tags) > 1 and example_tags[1].startswith("physician_agreed_category:"):
            subtheme = example_tags[1].replace("physician_agreed_category:", "")
        return theme, subtheme

    def get_name(self) -> str:
        return "HealthBench"


def download_file(url: str, out_path: str) -> None:
    logger.info("Downloading %s -> %s", url, out_path)
    urllib.request.urlretrieve(url, out_path)
    logger.info("Download complete")


def main():
    url = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"

    repo = os.environ.get("MEDEXPERT_REPO")
    if not repo:
        logger.error("Environment variable MEDEXPERT_REPO is not set")
        raise SystemExit(1)

    out_dir = os.path.join(repo, "factuality_omission_detection")
    os.makedirs(out_dir, exist_ok=True)

    filename = os.path.basename(urllib.parse.urlparse(url).path)
    download_path = os.path.join(out_dir, filename)
    download_file(url, download_path)

    # Process with standalone HealthBenchDataset
    ds = HealthBenchDataset(dataset_path=download_path)
    output_path = os.path.join(out_dir, "healthbench_icl_dataset.jsonl")
    logger.info("Writing %d processed records to %s", len(ds), output_path)
    with jsonlines.open(output_path, mode="w") as writer:
        writer.write_all(ds.dataset)
    logger.info("Done")


if __name__ == "__main__":
    main()