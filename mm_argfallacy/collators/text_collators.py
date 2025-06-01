from typing import Dict, Optional, List
from transformers import AutoTokenizer
from mm_argfallacy.utils.encoding import tokenize_text_with_context


class TextCollator:
    """
    Collator for text inputs using a transformer tokenizer (CPU-only).
    """

    def __init__(self, model_card: str, tokenizer_args: Optional[Dict] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_card)
        self.tokenizer_args = tokenizer_args or {}

    def __call__(self, inputs: List[str], context: Optional[List[str]] = None) -> Dict:
        input_ids, attention_mask, context_ids, context_mask = (
            tokenize_text_with_context(
                inputs=inputs,
                context=context,
                tokenizer=self.tokenizer,
                tokenizer_args=self.tokenizer_args,
            )
        )

        return {
            "inputs": input_ids,
            "input_mask": attention_mask,
            "context": context_ids,
            "context_mask": context_mask,
        }


class ConcatTextCollator:
    """
    Concatenates input and context text with a separator token.
    """

    def __init__(
        self,
        model_card: str,
        context_first: bool = True,
        tokenizer_args: Optional[Dict] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_card)
        self.tokenizer_args = tokenizer_args or {}
        self.context_first = context_first

        self.separator = self.tokenizer.sep_token or self.tokenizer.eos_token
        if self.separator is None:
            raise ValueError("Tokenizer has no separator or EOS token defined.")

        self.tokenizer.truncation_side = "left" if context_first else "right"

    def __call__(self, inputs: List[str], context: Optional[List[str]] = None) -> Dict:
        if context is not None and len(inputs) != len(context):
            raise ValueError("Length of inputs and context must match.")

        max_length = self.tokenizer_args.get(
            "max_length", self.tokenizer.model_max_length
        )
        sep_token_ids = self.tokenizer.encode(self.separator, add_special_tokens=False)
        sep_length = len(sep_token_ids)
        num_special = self.tokenizer.num_special_tokens_to_add(pair=False)

        concatenated_texts = []
        for i, input_text in enumerate(inputs):
            input_text = input_text or ""
            context_text = context[i] if context and context[i] else ""

            processed_context = context_text
            if not self.context_first and input_text and context_text:
                input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
                max_content_len = max_length - num_special
                available_context_len = max_content_len - len(input_ids) - sep_length

                if available_context_len <= 0:
                    processed_context = ""
                else:
                    context_ids = self.tokenizer.encode(
                        context_text, add_special_tokens=False
                    )
                    if len(context_ids) > available_context_len:
                        truncated_ids = context_ids[-available_context_len:]
                        processed_context = self.tokenizer.decode(
                            truncated_ids, skip_special_tokens=True
                        )

            parts = []
            if self.context_first:
                if processed_context:
                    parts.append(processed_context)
                if input_text:
                    parts.append(input_text)
            else:
                if input_text:
                    parts.append(input_text)
                if processed_context:
                    parts.append(processed_context)

            concatenated = self.separator.join(parts)
            concatenated_texts.append(concatenated)

        tokenized = self.tokenizer(
            concatenated_texts, padding=True, return_tensors="pt", **self.tokenizer_args
        )

        return {
            "inputs": tokenized["input_ids"],
            "input_mask": tokenized["attention_mask"],
        }
