# This material was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.
# PACIFIC NORTHWEST NATIONAL LABORATORY operated by BATTELLE for the UNITED STATES DEPARTMENT OF ENERGY under Contract DE-AC05-76RL01830.
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset, load_dataset
from scipy.special import softmax

from nukelm.utils import dir_path


PROJECT_DIR = Path(__file__).parents[3]
LOG = logging.getLogger(__name__)


def serve_documents(
    texts: List[str],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    pool_strategies: List[str],
    embed: bool = True,
    classify: bool = True,
    category_labels: Optional[List[str]] = None,
    use_cuda: bool = False,
) -> Dict[str, np.ndarray]:
    """Embed documents using various pooling strategies over transformers features.

    Args:
        texts (List[str]): [description]
        model (transformers.PreTrainedModel): [description]
        tokenizer (transformers.PreTrainedTokenizer): [description]
        pool_strategies (List[str]): [description]
        embed (bool): [Description]
        classify (bool): [Description]
        category_labels (Optional[List[str]]): [description]
        use_cuda (bool): [description]

    Raises:
        EnvironmentError: torch compiled without CUDA support
        ValueError: Unknown pooling strategy

    Returns:
        Dict[str, np.ndarray]: [description]
    """
    if use_cuda and not torch.cuda.is_available():
        raise OSError("Attempting to use CUDA but torch compiled without CUDA support.")

    result: Dict[str, np.ndarray] = dict()
    if not embed and not classify:
        LOG.warning("Neither `embed` nor `classify` set. Skipping processing.")
        return result

    inputs = tokenizer(
        texts, add_special_tokens=True, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    if use_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Aggregate embeddings, ignoring padding tokens
    if embed:
        features = outputs.hidden_states[-1]
        if use_cuda:
            inputs = {k: v.cpu() for k, v in inputs.items()}
            features = features.cpu()
        features = np.squeeze(features.numpy())
        if len(features.shape) == 2:
            features = features.reshape(1, *features.shape)
        batch_size, _, embedding_dim = features.shape
        result.update(
            {strategy: np.empty((batch_size, embedding_dim), dtype=features.dtype) for strategy in pool_strategies}
        )
        for i, (embedding, mask) in enumerate(zip(features, inputs["attention_mask"])):
            embedding = embedding[np.where(mask)[0], :]
            for strategy in pool_strategies:
                if strategy == "CLS":
                    result[strategy][i, :] = embedding[0]
                elif strategy == "MEAN":
                    result[strategy][i, :] = embedding.mean(axis=0)
                elif strategy == "MAX":
                    result[strategy][i, :] = embedding.max(axis=0)
                else:
                    raise ValueError(f"Unknown pooling strategy {strategy}")
        if batch_size == 1:
            result.update({strategy: result[strategy][0] for strategy in pool_strategies})

    # Convert logits to labels and scores
    if classify:
        logits = outputs.logits
        if use_cuda:
            logits = logits.cpu()
        if category_labels is None:
            category_labels = [f"Category {i}" for i in range(logits.shape[1])]
        scores = softmax(logits.numpy(), axis=1)
        scores = pd.DataFrame(scores, columns=category_labels)
        result["predicted-scores"] = np.array([row.to_dict() for _, row in scores.iterrows()])
        best_labels = scores.idxmax(axis=1).values
        result["predicted-label"] = best_labels

    return result


def main(
    model_path_or_name: str,
    input_path: str,
    output_path: str,
    tokenizer_path_or_name: Optional[str] = None,
    category_labels: Optional[List[str]] = None,
    batch_size: int = 5,
    use_cuda: bool = True,
) -> Dataset:
    """Extract features using a transformers model.

    Args:
        model_path_or_name (str): Name or path to transformers model.
        input_path (str): Path to CSV file containing a "text" column.
        output_path (str): Path to which to save dataset with extracted features.
        tokenizer_path_or_name (str, optional): Name or path to transformers tokenizer. Defaults to model_path_or_name.

        category_labels (Optional[List[str]]): Labels for predicted categories. Defaults to f"Category {i}".
        batch_size (int, optional): Number of samples to process at a time. Defaults to 5.

    Returns:
        datasets.Dataset: dataset enriched with embeddings and classifications.
    """
    LOG.debug(f"model_path_or_name={model_path_or_name}")
    LOG.debug(f"input_path={input_path}")
    LOG.debug(f"output_path={output_path}")
    LOG.debug(f"tokenizer_path_or_name={tokenizer_path_or_name}")
    LOG.debug(f"batch_size={batch_size}")
    config = transformers.AutoConfig.from_pretrained(model_path_or_name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path_or_name, config=config)
    if tokenizer_path_or_name is None:
        tokenizer_path_or_name = model_path_or_name
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path_or_name)

    dataset = load_dataset("csv", data_files=input_path)["train"]  # datasets puts a single data_files in "train"
    assert "text" in dataset.column_names

    try:
        category_labels = [config.id2label[i] for i in range(len(config.id2label))]
    except AttributeError:
        category_labels = None

    if use_cuda and torch.cuda.is_available():
        model.cuda()

    def _serve_documents(examples):
        return serve_documents(
            examples["text"],
            model=model,
            tokenizer=tokenizer,
            pool_strategies=["CLS", "MAX", "MEAN"],
            embed=True,
            classify=True,
            category_labels=category_labels,
            use_cuda=use_cuda and torch.cuda.is_available(),
        )

    dataset = dataset.map(
        _serve_documents,
        batched=True,
        batch_size=batch_size,
    )

    dataset.save_to_disk(output_path)


def endpoint():
    """CLI for model serving."""
    parser = argparse.ArgumentParser(description="Model serving pipeline")
    parser.add_argument(
        "--model-name",
        dest="model_name",
        default=None,
        type=str,
        help="Transformers model name",
    )
    parser.add_argument(
        "--model-path",
        dest="model_path",
        default=None,
        type=dir_path,
        help="Transformers model path",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        dest="output_path",
        type=str,
        help="Path to transformers model",
    )
    parser.add_argument(
        "-i",
        "--input-path",
        dest="input_path",
        type=str,
        default="data/01_raw/OSTI/binary_label_test_filtered_trimmed_1000.csv",
        help="path to CSV-formatted data",
    )
    parser.add_argument(
        "--tokenizer-name",
        dest="tokenizer_name",
        default=None,
        type=str,
        help="Transformers tokenizer name",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        default=5,
        type=int,
        help="Number of samples to process at one time",
    )
    args = parser.parse_args()
    input_path = (PROJECT_DIR / args.input_path).resolve().as_posix()
    output_path = (PROJECT_DIR / "data" / "07_model_output" / args.output_path).resolve().as_posix()
    assert (
        int(args.model_name is None) + int(args.model_path is None) == 1
    ), "Specify exactly one of `model-name` or `model-path`"
    model_name_or_path = args.model_name or (PROJECT_DIR / args.model_path).resolve().as_posix()

    main(model_name_or_path, input_path, output_path, args.tokenizer_name, args.batch_size)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(module)s.%(funcName)s.L%(lineno)d - %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    endpoint()
