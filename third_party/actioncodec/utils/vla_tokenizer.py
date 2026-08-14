import ast
import os
import re
from typing import Any, List, Literal

import numpy as np
from termcolor import cprint
from transformers import AutoModel, AutoProcessor, ProcessorMixin


class VisionLanguageActionProcessor(ProcessorMixin):
    """
    A unified wrapper processor that combines a vision-language (VL) processor with a
    custom action tokenizer. It supports five different modes for handling actions:

    1. "discrete": Expands the VL tokenizer's vocabulary with special action tokens
       and converts JSON-formatted action arrays into discrete action tokens.
    2. "mapped": Maps actions to existing vocabulary tokens without expanding the vocabulary.
    3. "numeric": Converts action arrays to raw numeric token sequences without
       vocabulary modifications.
    4. "discrete_bar": Adds special action tokens to the vocabulary and adds <|blk_bos|> and <|blk_eos|> tokens
    5. "identity": Parses action lists, strictly formats them to 2 decimal places in the text
       input (Encode), and parses them back to arrays with 2 decimal precision (Decode).

    Args:
        action_processor (`Any`):
            An object responsible for converting continuous action arrays into
            discrete integer token IDs.
        vl_processor (`ProcessorMixin`):
            A standard Hugging Face processor instance for a vision-language model.
        mode (`Literal["discrete", "mapped", "numeric", "discrete_bar", "identity"]`):
            The processing mode to use for actions.
            - "identity": Formats floats to 2 decimal places in input text and output arrays.
        **kwargs: Additional parameters for specific modes.
    """

    attributes = ["vl_processor"]
    VL_PROCESSOR_DIR = "vl_processor"
    ACTION_PROCESSOR_DIR = "action_processor"

    def __init__(
        self,
        action_processor: Any,
        vl_processor: ProcessorMixin,
        mode: Literal["discrete", "mapped", "numeric", "discrete_bar", "identity"] = "numeric",
        **kwargs,
    ):
        # This attribute is required by the parent class to verify the processor type.
        self.vl_processor_class = vl_processor.__class__.__name__
        self.mode = mode

        # Initialize token attributes before they are populated
        self.action_tokens = []
        self.action_token_ids = []
        self.vocab_shift = kwargs.get("vocab_shift", 0)

        # The parent constructor must be called.
        super().__init__(vl_processor, chat_template=vl_processor.chat_template)

        self.action_processor = action_processor

        # Initialize based on mode
        if mode == "discrete":
            self._expand_vl_vocab()
        elif mode == "discrete_bar":
            self._expand_vl_vocab(add_bar_tokens=True)
        elif mode == "mapped":
            self._setup_existing_vocab_tokens()
        # For "numeric" and "identity" mode, no vocabulary setup needed

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the processor to a directory.

        Creates the following directory structure:
            save_directory/
            ├── vl_processor/          # Vision-language processor files
            │   ├── tokenizer.json
            │   ├── preprocessor_config.json
            │   └── ...
            └── action_processor/      # Action processor files
                ├── config.json
                └── ...

        Args:
            save_directory (`str`):
                Directory path to save the processor components.
            **kwargs: Additional arguments passed to the underlying save_pretrained methods.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.vl_processor.save_pretrained(
            os.path.join(save_directory, self.VL_PROCESSOR_DIR), **kwargs
        )
        self.action_processor.save_pretrained(
            os.path.join(save_directory, self.ACTION_PROCESSOR_DIR), **kwargs
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load a processor from a local directory or Hugging Face Hub.

        For hub IDs: downloads entire repo to local cache first, then loads like local.
        For local paths: loads directly from subdirectories.

        Expected directory structure:
            pretrained_model_name_or_path/
            ├── vl_processor/          # Vision-language processor files
            │   ├── tokenizer.json
            │   ├── preprocessor_config.json
            │   └── ...
            └── action_processor/      # Action processor files
                ├── config.json (or processor_config.json)
                └── ...

        Args:
            pretrained_model_name_or_path (`str`):
                Local directory path or Hugging Face Hub model ID.
            **kwargs: Additional arguments:
                - action_processor_kwargs: Dict of kwargs for action processor loading
                  (default: {"trust_remote_code": True})
                - vl_processor_kwargs: Dict of kwargs for VL processor loading
                - token, revision, etc.: Hub download arguments

        Returns:
            VisionLanguageActionProcessor: The loaded processor instance.
        """
        from huggingface_hub import snapshot_download

        action_processor_kwargs = kwargs.pop("action_processor_kwargs", {"trust_remote_code": True})
        vl_processor_kwargs = kwargs.pop("vl_processor_kwargs", {})

        # Ensure trust_remote_code is set for action processor
        if "trust_remote_code" not in action_processor_kwargs:
            action_processor_kwargs["trust_remote_code"] = True

        # Check if it's a local path or a hub ID
        is_local = os.path.isdir(pretrained_model_name_or_path)

        if is_local:
            local_path = pretrained_model_name_or_path
        else:
            # Extract only valid snapshot_download kwargs
            snapshot_kwargs = {}
            for key in [
                "token",
                "use_auth_token",
                "revision",
                "repo_type",
                "cache_dir",
                "local_dir",
                "force_download",
                "proxies",
                "etag_timeout",
                "resume_download",
                "local_files_only",
            ]:
                if key in kwargs:
                    snapshot_kwargs[key] = kwargs.pop(key)
            # Download entire repo to local cache
            local_path = snapshot_download(repo_id=pretrained_model_name_or_path, **snapshot_kwargs)

        # Now load from local path (same logic for both cases)
        vl_processor = AutoProcessor.from_pretrained(
            os.path.join(local_path, cls.VL_PROCESSOR_DIR),
            **vl_processor_kwargs,
        )

        action_processor_path = os.path.join(local_path, cls.ACTION_PROCESSOR_DIR)
        if os.path.exists(os.path.join(action_processor_path, "processor_config.json")):
            action_processor = AutoProcessor.from_pretrained(
                action_processor_path, **action_processor_kwargs
            )
        else:
            action_processor = AutoModel.from_pretrained(
                action_processor_path, **action_processor_kwargs
            )

        return cls(action_processor=action_processor, vl_processor=vl_processor, **kwargs)

    def _expand_vl_vocab(self, add_bar_tokens: bool = False):
        """
        Expand the vision-language tokenizer vocabulary with action tokens.

        Creates special tokens of the form `<|action_0|>`, `<|action_1|>`, ..., up to
        `vocab_size - 1` tokens. Optionally adds block boundary markers for
        discrete_bar mode.

        Args:
            add_bar_tokens (`bool`):
                If True, adds `<|blk_bos|>` and `<|blk_eos|>` tokens for block-wise
                action sequence modeling.

        Note:
            After calling this method, you must resize the model embeddings:
            `model.resize_token_embeddings(len(processor.tokenizer))`
        """
        action_vocab_size = self.action_processor.vocab_size
        action_specific_tokens = [f"<|action_{i}|>" for i in range(action_vocab_size)]

        if add_bar_tokens:
            action_specific_tokens.append("<|blk_bos|>")
            action_specific_tokens.append("<|blk_eos|>")

        self.action_tokens = action_specific_tokens
        self.vl_processor.tokenizer.add_tokens(action_specific_tokens, special_tokens=True)
        print(f"Vocabulary expanded. New size: {len(self.vl_processor.tokenizer)}")
        cprint(
            "Note: Remember to resize model embeddings after loading, e.g., model.resize_token_embeddings(len(processor.tokenizer))",
            "yellow",
        )
        self.action_token_ids = self.vl_processor.tokenizer.convert_tokens_to_ids(
            action_specific_tokens
        )

    def _setup_existing_vocab_tokens(self):
        """
        Set up action token mapping using existing vocabulary tokens.

        Instead of expanding the vocabulary, this method maps action token IDs to
        a range of existing vocabulary tokens. This is useful when you cannot modify
        the model's embedding layer.

        The mapping uses tokens starting from index `max(1000, vocab_size - vocab_shift - action_vocab_size)`
        to avoid special tokens and common tokens at the beginning of the vocabulary.

        Raises:
            ValueError: If there isn't enough vocabulary space for the required tokens.

        Note:
            Requires `vocab_shift` to be set in the constructor kwargs.
        """
        action_vocab_size = self.action_processor.vocab_size
        vocab_size = len(self.vl_processor.tokenizer)
        start_idx = max(1000, vocab_size - self.vocab_shift - action_vocab_size)
        end_idx = start_idx + action_vocab_size

        if end_idx > vocab_size - self.vocab_shift:
            raise ValueError(
                f"Not enough vocabulary space. Need {action_vocab_size} tokens, "
                f"but only {vocab_size - self.vocab_shift - start_idx} available."
            )

        self.action_token_ids = list(range(start_idx, end_idx))
        self.vocab_to_action_mapping = {vid: aid for aid, vid in enumerate(self.action_token_ids)}
        print(f"Using existing vocabulary tokens {start_idx} to {end_idx - 1} for actions")
        print(f"Original vocab size: {vocab_size} (unchanged)")

    def change_mode(
        self,
        new_mode: Literal["discrete", "mapped", "numeric", "discrete_bar", "identity"],
        **kwargs,
    ):
        if new_mode == self.mode:
            print(f"Already in {new_mode} mode. No change needed.")
            return

        print(f"Changing mode from {self.mode} to {new_mode}")

        self.action_tokens = []
        self.action_token_ids = []
        if hasattr(self, "vocab_to_action_mapping"):
            delattr(self, "vocab_to_action_mapping")

        self.mode = new_mode
        if new_mode == "mapped":
            self.vocab_shift = kwargs.get("vocab_shift", 0)

        if new_mode == "discrete":
            cprint("WARNING: Switching to discrete mode will expand the vocabulary!", "yellow")
            self._expand_vl_vocab(add_bar_tokens=False)
        elif new_mode == "discrete_bar":
            cprint("WARNING: Switching to discrete mode will expand the vocabulary!", "yellow")
            self._expand_vl_vocab(add_bar_tokens=True)
        elif new_mode == "mapped":
            self._setup_existing_vocab_tokens()

        print(f"Successfully changed to {new_mode} mode")

    def _get_action_token(self, action: np.ndarray) -> List[int]:
        return self.action_processor.encode(action)

    def _get_action_from_token(self, token: List[int]) -> np.ndarray:
        return self.action_processor.decode(token)

    def __call__(self, *args, **kwargs):
        """
        Process inputs, converting action arrays in JSON strings into token representations.

        This method intercepts text inputs containing action dictionaries (format:
        `{'action': [[...]]}`) and converts them based on the current mode:

        - **discrete**: Converts to `<|action_0|><|action_1|>...` token sequence
        - **discrete_bar**: Wraps action tokens with `<|blk_bos|>...<|blk_eos|>`
        - **mapped**: Converts to existing vocabulary tokens via mapping
        - **numeric**: Converts to raw integer sequence `{'action': [0,1,2,...]}`
        - **identity**: Rounds all values to 2 decimal places (no tokenization)

        Args:
            *args: Positional arguments passed to the underlying VL processor.
            **kwargs: Keyword arguments. Special keys:
                - text: List of text strings to process
                - action_processor_kwargs: Dict of kwargs for action encoding

        Returns:
            Processed outputs from the underlying VL processor (typically dict with
            input_ids, attention_mask, pixel_values, etc.)
        """
        text_inputs = kwargs.get("text", None)
        action_processor_kwargs = kwargs.get("action_processor_kwargs", {})

        if text_inputs is None or not isinstance(text_inputs, list):
            return self.vl_processor(*args, **kwargs)

        processed_texts = list(text_inputs)

        # Regex to find a dictionary-like string with an 'action' key.
        action_pattern = re.compile(r"(\{'action':\s*\[.*?\]\})")

        for i, text in enumerate(processed_texts):
            match = action_pattern.search(text)
            if not match:
                continue

            try:
                action_str = match.group(1)
                action_dict = ast.literal_eval(action_str)
                continuous_action = action_dict.get("action")

                if continuous_action is None:
                    continue

                # Prepare common NumPy array for modes that need it
                actions_np = np.array([continuous_action], dtype=np.float32)

                if self.mode == "identity":
                    # For identity mode, we don't tokenize.
                    # We strictly round the input array to 2 decimal places and reform the string.
                    # continuous_action is typically a list of lists.
                    rounded_action = np.round(actions_np[0], 2).tolist()
                    rounded_action = [[round(a, 2) for a in step] for step in rounded_action]
                    new_action_str = f"{{'action': {str(rounded_action)}}}"

                else:
                    # For all other modes, we need the action processor
                    action_token_ids = self.action_processor.encode(
                        actions_np, **action_processor_kwargs
                    )
                    flat_token_ids = action_token_ids[0]

                    if self.mode == "discrete":
                        action_tokens = [f"<|action_{tid}|>" for tid in flat_token_ids]
                        action_string = "".join(action_tokens)
                        new_action_str = f"{{'action': '{action_string}'}}"
                    elif self.mode == "mapped":
                        mapped_token_ids = [self.action_token_ids[tid] for tid in flat_token_ids]
                        action_string = self.vl_processor.tokenizer.decode(mapped_token_ids)
                        new_action_str = f"{{'action': '{action_string}'}}"
                    elif self.mode == "discrete_bar":
                        action_tokens = [f"<|action_{tid}|>" for tid in flat_token_ids]
                        action_string = "".join(action_tokens)
                        new_action_str = f"{{'action': '<|blk_bos|>{action_string}<|blk_eos|>'}}"
                    else:  # numeric mode
                        new_action_str = f"{{'action':[{','.join(map(str, flat_token_ids))}]}}"

                # Replace the original action block with the new formatted/tokenized one
                processed_texts[i] = text.replace(action_str, new_action_str, 1)

            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse or process action in text: '{text}'. Error: {e}")
                continue

        kwargs["text"] = processed_texts
        return self.vl_processor(**kwargs)

    def _decode_actions(
        self,
        texts: List[str],
        action_processor_kwargs: dict = {},
        precision: int = 8,
        actions_only: bool = False,
    ) -> List[Any]:
        """
        Decode action token sequences within JSON strings back to action arrays.

        Each mode uses specific regex patterns to identify and decode actions:

        - **discrete**: Matches `{'action': '<|action_0|><|action_1|>...'}`,
          extracts token IDs from `<|action_N|>` patterns
        - **discrete_bar**: Matches `{'action': '<|blk_bos|>...<|blk_eos|>'}`,
          same extraction as discrete
        - **mapped**: Matches `{'action': '...'}`, decodes text to vocab IDs,
          then maps back to action token IDs
        - **numeric**: Matches `{'action': [0,1,2,...]}`, parses integer list
        - **identity**: Matches `{'action': [[...]]}`, parses and rounds to 2 decimals

        Args:
            texts (`List[str]`): List of text strings containing encoded actions.
            action_processor_kwargs (`dict`): Kwargs for the action processor's decode method.
            precision (`int`): Number of decimal places for rounding (default: 8).
                Note: identity mode always uses 2 decimal places.
            actions_only (`bool`): If True, return only the decoded action lists
                instead of the full text with actions replaced.

        Returns:
            `List[Any]`: If actions_only is False, returns texts with decoded actions.
            If actions_only is True, returns lists of decoded action values.
        """
        is_str_input = isinstance(texts, str)
        if is_str_input:
            texts = [texts]

        processed_outputs = []
        for text in texts:
            decoded_action_list = None

            if self.mode == "discrete":
                action_pattern = re.compile(r"\{'action':\s*'([^']+)'\}")
                id_extractor_pattern = re.compile(r"<\|action_(\d+)\|>")
                matches = list(action_pattern.finditer(text))
                for match in reversed(matches):
                    action_tokens_str = match.group(1)
                    try:
                        token_ids = [
                            int(i) for i in id_extractor_pattern.findall(action_tokens_str)
                        ]
                        if not token_ids:
                            continue
                        decoded_action_np = self.action_processor.decode(
                            [token_ids], **action_processor_kwargs
                        )
                        if not isinstance(decoded_action_np, np.ndarray):
                            decoded_action_np = decoded_action_np[0]
                        decoded_action_list = np.round(decoded_action_np[0], precision).tolist()
                        new_action_str = f"{{'action': {str(decoded_action_list)}}}"
                        text = text[: match.start()] + new_action_str + text[match.end() :]
                    except Exception as e:
                        print(f"Warning: Could not decode discrete action tokens. Error: {e}")

            elif self.mode == "discrete_bar":
                action_pattern = re.compile(r"\{'action':\s*'<\|blk_bos\|>([^']+)<\|blk_eos\|>'\}")
                id_extractor_pattern = re.compile(r"<\|action_(\d+)\|>")
                matches = list(action_pattern.finditer(text))
                for match in reversed(matches):
                    action_tokens_str = match.group(1)
                    try:
                        token_ids = [
                            int(i) for i in id_extractor_pattern.findall(action_tokens_str)
                        ]
                        if not token_ids:
                            continue
                        decoded_action_np = self.action_processor.decode(
                            [token_ids], **action_processor_kwargs
                        )
                        if not isinstance(decoded_action_np, np.ndarray):
                            decoded_action_np = decoded_action_np[0]
                        decoded_action_list = np.round(decoded_action_np[0], precision).tolist()
                        new_action_str = f"{{'action': {str(decoded_action_list)}}}"
                        text = text[: match.start()] + new_action_str + text[match.end() :]
                    except Exception as e:
                        print(f"Warning: Could not decode discrete bar action tokens. Error: {e}")

            elif self.mode == "mapped":
                action_pattern = re.compile(r"\{'action':\s*'([^']+)'\}")
                matches = list(action_pattern.finditer(text))
                for match in reversed(matches):
                    token_string = match.group(1)
                    try:
                        token_ids = self.vl_processor.tokenizer.encode(
                            token_string, add_special_tokens=False
                        )
                        action_token_ids = [
                            self.vocab_to_action_mapping[tid]
                            for tid in token_ids
                            if tid in self.vocab_to_action_mapping
                        ]
                        if not action_token_ids:
                            continue
                        decoded_action_np = self.action_processor.decode(
                            [action_token_ids], **action_processor_kwargs
                        )
                        if not isinstance(decoded_action_np, np.ndarray):
                            decoded_action_np = decoded_action_np[0]
                        decoded_action_list = np.round(decoded_action_np[0], precision).tolist()
                        new_action_str = f"{{'action': {str(decoded_action_list)}}}"
                        text = text[: match.start()] + new_action_str + text[match.end() :]
                    except Exception as e:
                        print(f"Warning: Could not decode mapped action tokens. Error: {e}")

            elif self.mode == "identity":
                # Regex for list format: "{'action': [ ... ]}"
                action_pattern = re.compile(r"\{'action':\s*(\[.*?\])\}", re.DOTALL)
                matches = list(action_pattern.finditer(text))
                for match in reversed(matches):
                    list_content_str = match.group(1)
                    try:
                        action_list = ast.literal_eval(list_content_str)
                        if not isinstance(action_list, list):
                            continue
                        # Strict 2 decimal places constraint
                        decoded_action_np = np.array(action_list)
                        decoded_action_list = np.round(decoded_action_np, 2).tolist()
                        new_action_str = f"{{'action': {str(decoded_action_list)}}}"
                        text = text[: match.start()] + new_action_str + text[match.end() :]
                    except Exception as e:
                        print(f"Warning: Could not parse identity action list. Error: {e}")

            else:  # numeric
                action_pattern = re.compile(r"\{'action':\s*\[([\d,]+)\]\}")
                matches = list(action_pattern.finditer(text))
                for match in reversed(matches):
                    token_ids_str = match.group(1)
                    try:
                        token_ids = [int(i) for i in token_ids_str.split(",")]
                        decoded_action_np = self.action_processor.decode(
                            [token_ids], **action_processor_kwargs
                        )
                        if not isinstance(decoded_action_np, np.ndarray):
                            decoded_action_np = decoded_action_np[0]
                        decoded_action_list = np.round(decoded_action_np[0], precision).tolist()
                        new_action_str = f"{{'action': {str(decoded_action_list)}}}"
                        text = text[: match.start()] + new_action_str + text[match.end() :]
                    except Exception as e:
                        print(f"Warning: Could not decode numeric action tokens. Error: {e}")

            if actions_only:
                processed_outputs.append(decoded_action_list)
            else:
                processed_outputs.append(text)

        return processed_outputs[0] if is_str_input else processed_outputs

    def decode(
        self,
        *args,
        decode_actions: bool = True,
        precision: int = 8,
        actions_only: bool = False,
        action_processor_kwargs: dict = {},
        **kwargs,
    ):
        decoded_outputs = self.vl_processor.decode(*args, **kwargs)
        if decode_actions:
            decoded_outputs = self._decode_actions(
                decoded_outputs,
                action_processor_kwargs=action_processor_kwargs,
                precision=precision,
                actions_only=actions_only,
            )
        return decoded_outputs

    def batch_decode(
        self,
        *args,
        decode_actions: bool = True,
        precision: int = 8,
        actions_only: bool = False,
        action_processor_kwargs: dict = {},
        **kwargs,
    ):
        decoded_outputs = self.vl_processor.batch_decode(*args, **kwargs)
        if decode_actions:
            decoded_outputs = self._decode_actions(
                decoded_outputs,
                precision=precision,
                actions_only=actions_only,
                action_processor_kwargs=action_processor_kwargs,
            )
        return decoded_outputs

    def __getattr__(self, name: str) -> Any:
        if "vl_processor" not in self.__dict__:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        try:
            return getattr(self.vl_processor, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


if __name__ == "__main__":
    """
    Test script for VisionLanguageActionProcessor.

    Usage:
        python utils/vla_tokenizer.py

        # Test specific mode only
        TEST_MODES=discrete,numeric python utils/vla_tokenizer.py
    """
    import os

    class SimpleBinActionProcessor:
        """
        Simple bin-based action processor for testing.

        Clips actions to [-1, 1] and discretizes into uniform bins.
        """

        def __init__(self, vocab_size: int = 1000):
            self.vocab_size = vocab_size

        def encode(self, actions: np.ndarray, **kwargs) -> list:
            """
            Encode continuous actions to bin indices.

            Args:
                actions: np.ndarray of shape (batch, horizon, action_dim)

            Returns:
                List of token ID lists, one per batch item.
            """
            # Clip to [-1, 1]
            clipped = np.clip(actions, -1.0, 1.0)
            # Map to [0, vocab_size-1]
            normalized = (clipped + 1.0) / 2.0  # [0, 1]
            token_ids = (normalized * (self.vocab_size - 1)).astype(np.int64)
            # Flatten horizon and action_dim into single sequence
            batch_size = token_ids.shape[0]
            return [token_ids[i].flatten().tolist() for i in range(batch_size)]

        def decode(self, tokens: list, **kwargs) -> np.ndarray:
            """
            Decode bin indices back to continuous actions.

            Args:
                tokens: List of token ID lists

            Returns:
                np.ndarray of shape (batch, horizon, action_dim)
            """
            results = []
            for token_list in tokens:
                token_arr = np.array(token_list, dtype=np.float32)
                # Map back to [-1, 1]
                normalized = token_arr / (self.vocab_size - 1)
                continuous = normalized * 2.0 - 1.0
                results.append(continuous)
            return np.array(results)

    # Configuration
    VL_PROCESSOR_PATH = os.environ.get(
        "VL_PROCESSOR_PATH", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    )
    TEST_MODES = os.environ.get(
        "TEST_MODES", "mapped,numeric,identity,discrete,discrete_bar"
    ).split(",")

    print("=" * 60)
    print("VisionLanguageActionProcessor Test Suite")
    print("=" * 60)
    print(f"VL Processor: {VL_PROCESSOR_PATH}")
    print(f"Test Modes: {TEST_MODES}")
    print()

    # Create simple bin action processor
    action_processor = SimpleBinActionProcessor(vocab_size=1000)
    print("Action processor: SimpleBinActionProcessor(vocab_size=1000)")

    # Load VL processor
    print("Loading VL processor...")
    vl_processor = AutoProcessor.from_pretrained(VL_PROCESSOR_PATH, trust_remote_code=True)
    print("VL processor loaded successfully.\n")

    def create_test_messages(action: np.ndarray) -> list:
        """Create test messages with action data."""
        return [
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "{'action':"
                            + f"{[[round(a, 5) for a in step] for step in action.tolist()]}"
                            + "}",
                        }
                    ],
                },
            ]
        ]

    def test_mode(processor, mode: str, messages: list, action: np.ndarray) -> bool:
        """Test a specific mode and return True if successful."""
        # Apply chat template
        messages_processed = processor.apply_chat_template(messages, add_generation_prompt=False)
        print(f"  Input text (truncated): {messages_processed[0][:100]}...")

        # Encode
        processed_batch = processor(
            text=messages_processed,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )

        # Show encoded content
        input_ids = processed_batch["input_ids"][0].tolist()
        print(f"  Encoded token IDs (len={len(input_ids)}): {str(input_ids[:20])[:-1]}...")

        # Decode
        decoded_outputs_decode_action = processor.batch_decode(
            processed_batch["input_ids"],
            decode_actions=True,
            precision=4,
            actions_only=False,
        )

        # Show decoded output
        output_decode_action = decoded_outputs_decode_action[0]
        print(f"  Decoded output with action decoding: {output_decode_action[:100]}...")

        # Decode
        decoded_outputs_no_decode_action = processor.batch_decode(
            processed_batch["input_ids"],
            decode_actions=False,
            precision=4,
            actions_only=False,
        )

        # Show decoded output
        output_no_decode_action = decoded_outputs_no_decode_action[0]
        print(f"  Decoded output without action decoding: {output_no_decode_action[:100]}...")

        return True

    # Run tests
    np.random.seed(42)  # For reproducibility
    action = np.random.randn(20, 7) * 0.5  # Scale down to mostly fit in [-1, 1]
    messages = create_test_messages(action)
    print(f"Test action shape: {action.shape}")
    print(f"Test action range: [{action.min():.3f}, {action.max():.3f}]\n")

    for mode in TEST_MODES:
        mode = mode.strip()
        print(f"\n{'=' * 60}")
        print(f"Testing mode: {mode}")
        print("=" * 60)

        kwargs = {}
        if mode == "mapped":
            kwargs["vocab_shift"] = 100

        try:
            processor = VisionLanguageActionProcessor(
                action_processor, vl_processor, mode=mode, **kwargs
            )
            test_mode(processor, mode, messages, action)
            print(f"  [PASS] {mode} mode test completed")
        except Exception as e:
            import traceback

            print(f"  [FAIL] {mode} mode test failed: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test suite completed")
    print("=" * 60)
