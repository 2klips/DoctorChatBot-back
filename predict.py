# Generate answer from input text
# Author: MEDIAZEN AIMZ R&D Group NLP Team

import re
import os
import sys
import random
import argparse
import torch

from pathlib import Path
from typing import List, Dict, Any
sys.path.append(
    str(Path(__file__).parent.parent)
)
from transformers import T5ForConditionalGeneration, T5TokenizerFast


def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = T5TokenizerFast.from_pretrained(model_path)
    return model, tokenizer

def main(args):
    model, tokenizer = load_model(args.model_path)
    model.eval()

    with torch.no_grad():
        while True:
            input_text = input("input text: ")
            if input_text in ["quit", "exit", "q", "종료", "나가기"]:
                break

            input_ids = tokenizer(
                input_text,
                return_tensors='pt',
            ).input_ids.to(model.device)

            output = model.generate(
                input_ids,
                max_length=1024,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                top_p=1,
            )

            decoded_output = tokenizer.decode(
                output[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            print(f"모델 출력 : {decoded_output}")
            print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="model path",
    )
    args = parser.parse_args()

    main(args)
