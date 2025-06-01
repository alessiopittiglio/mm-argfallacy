from typing import Dict, Optional, List, Tuple


def tokenize_text(
    inputs: List[Optional[str]],
    tokenizer,
    tokenizer_args: Optional[Dict] = None,
):
    """
    Encodes input texts using a tokenizer. Empty strings are substituted for None.

    Args:
        inputs: List of input strings.
        tokenizer: HuggingFace tokenizer.
        tokenizer_args: Additional arguments for tokenizer.

    Returns:
        Tokenized input dictionary with input IDs and attention mask.
    """
    tokenizer_args = tokenizer_args or {}
    cleaned_inputs = [text or "" for text in inputs]

    return tokenizer(
        cleaned_inputs,
        padding=True,
        return_tensors="pt",
        **tokenizer_args,
    )


def tokenize_text_with_context(
    inputs: List[str],
    tokenizer,
    tokenizer_args: Optional[Dict] = None,
    context: Optional[List[str]] = None,
) -> Tuple:
    """
    Encodes input and context texts into token IDs and attention masks.

    Returns:
        Tuple of input IDs, input masks, context IDs, context masks.
    """
    tok_inputs = tokenize_text(inputs, tokenizer, tokenizer_args)

    tok_context = {"input_ids": None, "attention_mask": None}
    if context is not None:
        tok_context = tokenize_text(context, tokenizer, tokenizer_args)

    return (
        tok_inputs["input_ids"],
        tok_inputs["attention_mask"],
        tok_context["input_ids"],
        tok_context["attention_mask"],
    )
