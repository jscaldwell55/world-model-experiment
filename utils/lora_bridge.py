"""
LoRA Bridge for World Model Graduation

Provides LoRA fine-tuning infrastructure for graduating world model knowledge
from playbook observations into model weights.

Supports small models for local testing:
- Qwen/Qwen2.5-0.5B (recommended for testing)
- Qwen/Qwen2.5-1.5B
- Any HuggingFace causal LM

Usage:
    from utils.lora_bridge import LoRABridge

    bridge = LoRABridge(model_name="Qwen/Qwen2.5-0.5B")
    bridge.train(training_pairs, output_dir="./lora_adapters/hot_pot")
    response = bridge.generate("What is the heating rate at HIGH power?")
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """Configuration for LoRA training"""
    r: int = 16                      # LoRA rank
    lora_alpha: int = 32             # LoRA alpha scaling
    lora_dropout: float = 0.05       # Dropout for LoRA layers
    target_modules: List[str] = None # Target modules (auto-detected if None)
    bias: str = "none"               # Bias handling


@dataclass
class TrainingConfig:
    """Configuration for training"""
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.1
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 1
    fp16: bool = False               # Use fp16 (if GPU supports)
    logging_steps: int = 10
    save_steps: int = 100


class LoRABridge:
    """
    LoRA training and inference bridge for world model graduation.

    Handles:
    - Model loading with appropriate configuration
    - LoRA adapter configuration
    - Training loop with proper formatting
    - Inference with/without adapters
    - Proper baseline capture before training (to avoid contamination)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        device: str = "auto",
        load_in_8bit: bool = False,
        lora_config: Optional[LoRAConfig] = None,
        training_config: Optional[TrainingConfig] = None
    ):
        """
        Initialize LoRA Bridge.

        Args:
            model_name: HuggingFace model name or path
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            load_in_8bit: Whether to load model in 8-bit quantization
            lora_config: LoRA configuration
            training_config: Training configuration
        """
        self.model_name = model_name
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.lora_config = lora_config or LoRAConfig()
        self.training_config = training_config or TrainingConfig()

        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self._initialized = False
        self._baseline_captured = False
        self._cached_baselines = {}  # Store pre-training baseline responses

    def _lazy_init(self):
        """Lazy initialization of model and tokenizer"""
        if self._initialized:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            )

        print(f"Loading model: {self.model_name}")

        # Determine device
        if self.device == "auto":
            if torch.cuda.is_available():
                device_map = "auto"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_map = {"": "mps"}
            else:
                device_map = {"": "cpu"}
        else:
            device_map = {"": self.device}

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": device_map,
        }

        if self.load_in_8bit:
            try:
                import bitsandbytes
                model_kwargs["load_in_8bit"] = True
            except ImportError:
                print("Warning: bitsandbytes not available, loading in fp32")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        self._initialized = True
        print(f"Model loaded on device: {self.model.device}")

    def _setup_lora(self):
        """Setup LoRA adapter on the model"""
        self._lazy_init()

        try:
            from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError(
                "peft is required for LoRA. "
                "Install with: pip install peft"
            )

        # Determine target modules based on model architecture
        if self.lora_config.target_modules is None:
            # Common target modules for different architectures
            if "qwen" in self.model_name.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            elif "llama" in self.model_name.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            elif "gpt" in self.model_name.lower():
                target_modules = ["c_attn", "c_proj"]
            else:
                # Default: try common attention modules
                target_modules = ["q_proj", "v_proj"]
        else:
            target_modules = self.lora_config.target_modules

        peft_config = PeftLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            target_modules=target_modules,
            bias=self.lora_config.bias
        )

        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()

        return self.peft_model

    def _format_training_example(self, pair: Dict) -> str:
        """
        Format a training pair as a prompt-response string.

        Args:
            pair: Dict with 'instruction' and 'response' keys

        Returns:
            Formatted string for training
        """
        instruction = pair.get('instruction', '')
        response = pair.get('response', '')

        # Format as instruction-following template
        return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

    def _prepare_dataset(self, training_pairs: List[Dict]):
        """
        Prepare dataset for training.

        Args:
            training_pairs: List of dicts with 'instruction' and 'response'

        Returns:
            HuggingFace Dataset object
        """
        try:
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "datasets is required. "
                "Install with: pip install datasets"
            )

        # Format all pairs
        formatted_texts = [
            self._format_training_example(pair)
            for pair in training_pairs
        ]

        # Create dataset
        dataset = Dataset.from_dict({"text": formatted_texts})

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.training_config.max_seq_length,
                padding="max_length"
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        return tokenized_dataset

    def train(
        self,
        training_pairs: List[Dict],
        output_dir: str,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict:
        """
        Train LoRA adapter on training pairs.

        Args:
            training_pairs: List of dicts with 'instruction' and 'response'
            output_dir: Directory to save adapter
            resume_from_checkpoint: Optional checkpoint to resume from

        Returns:
            Training metrics dict
        """
        self._lazy_init()
        self._setup_lora()

        # Import with TF disabled to avoid Keras version conflicts
        import os
        os.environ["USE_TF"] = "0"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        try:
            from transformers import (
                Trainer,
                TrainingArguments,
                DataCollatorForLanguageModeling
            )
        except ImportError:
            raise ImportError(
                "transformers Trainer is required. "
                "Install with: pip install transformers"
            )

        print(f"\nPreparing dataset with {len(training_pairs)} training pairs...")
        dataset = self._prepare_dataset(training_pairs)

        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            warmup_ratio=self.training_config.warmup_ratio,
            fp16=self.training_config.fp16,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            save_total_limit=2,
            report_to="none",  # Disable wandb/tensorboard
            remove_unused_columns=False,
        )

        # Create trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        print(f"\nStarting training...")
        print(f"  Epochs: {self.training_config.num_epochs}")
        print(f"  Batch size: {self.training_config.batch_size}")
        print(f"  Learning rate: {self.training_config.learning_rate}")
        print(f"  Output: {output_dir}")

        # Train
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save adapter
        print(f"\nSaving adapter to {output_dir}")
        self.peft_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save training config
        config_path = Path(output_dir) / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'lora_config': {
                    'r': self.lora_config.r,
                    'lora_alpha': self.lora_config.lora_alpha,
                    'lora_dropout': self.lora_config.lora_dropout,
                },
                'training_config': {
                    'num_epochs': self.training_config.num_epochs,
                    'batch_size': self.training_config.batch_size,
                    'learning_rate': self.training_config.learning_rate,
                },
                'num_training_pairs': len(training_pairs),
            }, f, indent=2)

        return train_result.metrics

    def load_adapter(self, adapter_path: str):
        """
        Load a trained LoRA adapter.

        Args:
            adapter_path: Path to saved adapter directory
        """
        self._lazy_init()

        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("peft is required. Install with: pip install peft")

        print(f"Loading adapter from {adapter_path}")
        self.peft_model = PeftModel.from_pretrained(
            self.model,
            adapter_path
        )
        print("Adapter loaded successfully")

    def capture_baseline(
        self,
        test_questions: List[Dict],
        max_new_tokens: int = 150,
        temperature: float = 0.3
    ) -> Dict[tuple, str]:
        """
        Capture baseline responses BEFORE any training.

        IMPORTANT: Call this AFTER _lazy_init() but BEFORE _setup_lora() or train().
        This captures the true baseline model responses before any contamination.

        Args:
            test_questions: List of {"domain": str, "question": str}
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Dict mapping (domain, question) -> response
        """
        if self.peft_model is not None:
            raise RuntimeError(
                "Cannot capture baseline after adapter creation. "
                "Call capture_baseline() before train() or _setup_lora()."
            )

        self._lazy_init()

        print(f"\nCapturing baseline responses for {len(test_questions)} questions...")

        for item in test_questions:
            domain = item["domain"]
            question = item["question"]
            key = (domain, question)

            # Generate using raw base model
            response = self._generate_raw(
                question,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            self._cached_baselines[key] = response

        self._baseline_captured = True
        print(f"Captured {len(self._cached_baselines)} baseline responses")

        return self._cached_baselines.copy()

    def _generate_raw(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """
        Generate using the raw base model (no adapter).
        Internal method used for baseline capture.
        """
        import torch

        # Always use self.model directly (not peft_model)
        model = self.model

        # Format prompt
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.training_config.max_seq_length
        )

        # Move to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response
        if "### Response:" in full_output:
            response = full_output.split("### Response:")[-1].strip()
        else:
            response = full_output[len(formatted_prompt):].strip()

        return response

    def get_cached_baseline(self, domain: str, question: str) -> Optional[str]:
        """
        Get a cached baseline response.

        Args:
            domain: Domain name
            question: Question text

        Returns:
            Cached baseline response, or None if not captured
        """
        return self._cached_baselines.get((domain, question))

    def has_baseline_for(self, domain: str, question: str) -> bool:
        """Check if baseline was captured for a specific question."""
        return (domain, question) in self._cached_baselines

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        use_adapter: bool = True,
        return_full: bool = False
    ) -> str:
        """
        Generate response for a prompt.

        Args:
            prompt: Input prompt/instruction
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_adapter: If True, use LoRA adapter; if False, use base model
                         NOTE: After training, use_adapter=False uses PEFT's
                         disable_adapter() context to get true baseline behavior.
            return_full: If True, return full output including prompt

        Returns:
            Generated response string
        """
        self._lazy_init()

        import torch

        # Format prompt
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.training_config.max_seq_length
        )

        # Determine which model/mode to use
        if use_adapter and self.peft_model is not None:
            # Use PEFT model with adapter enabled
            model = self.peft_model
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        elif not use_adapter and self.peft_model is not None:
            # Use PEFT model but with adapter DISABLED
            # This properly bypasses the LoRA weights
            model = self.peft_model
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                with model.disable_adapter():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
        else:
            # No PEFT model yet, use base model directly
            model = self.model
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if return_full:
            return full_output

        # Extract just the response part
        if "### Response:" in full_output:
            response = full_output.split("### Response:")[-1].strip()
        else:
            response = full_output[len(formatted_prompt):].strip()

        return response

    def compare_with_baseline(
        self,
        test_prompts: List[str],
        max_new_tokens: int = 256
    ) -> List[Dict]:
        """
        Compare adapter responses with baseline model responses.

        Args:
            test_prompts: List of test prompts
            max_new_tokens: Maximum tokens to generate

        Returns:
            List of comparison results
        """
        results = []

        for prompt in test_prompts:
            # Generate with adapter
            adapter_response = self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                use_adapter=True
            )

            # Generate without adapter (baseline)
            baseline_response = self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                use_adapter=False
            )

            results.append({
                'prompt': prompt,
                'adapter_response': adapter_response,
                'baseline_response': baseline_response,
                'responses_differ': adapter_response != baseline_response
            })

        return results


def create_lora_bridge(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    lora_r: int = 16,
    lora_alpha: int = 32,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 3e-4
) -> LoRABridge:
    """
    Convenience function to create a configured LoRA bridge.

    Args:
        model_name: HuggingFace model name
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate

    Returns:
        Configured LoRABridge instance
    """
    lora_config = LoRAConfig(r=lora_r, lora_alpha=lora_alpha)
    training_config = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    return LoRABridge(
        model_name=model_name,
        lora_config=lora_config,
        training_config=training_config
    )
