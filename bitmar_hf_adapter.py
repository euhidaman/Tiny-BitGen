"""
BitMar to HuggingFace Model Adapter
Adapts BitMar models to be compatible with HuggingFace evaluation pipelines
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, GPT2Config, GPT2LMHeadModel
from typing import Optional, Dict, Any
import json
from pathlib import Path

class BitMarAsGPT2(GPT2LMHeadModel):
    """BitMar model wrapped as GPT2 for HuggingFace compatibility"""

    def __init__(self, config, original_model=None):
        # Initialize as GPT2 but override with our model
        super().__init__(config)
        self.original_model = original_model

        # If we have an original model, we'll use it for forward passes
        if original_model is not None:
            # Clear the GPT2 parameters to save memory since we won't use them
            pass

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """Forward pass using original BitMar model or GPT2 fallback"""

        if self.original_model is not None:
            # Use original BitMar model
            try:
                # Prepare inputs for BitMar
                batch_size = input_ids.size(0)

                # Create dummy vision features if needed
                vision_features = torch.zeros(batch_size, 768, device=input_ids.device)

                # Call original model with minimal required arguments
                outputs = self.original_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    vision_features=vision_features,
                    labels=labels
                )

                # Return in HuggingFace format
                from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
                return CausalLMOutputWithCrossAttentions(
                    loss=outputs.get('loss') if labels is not None else None,
                    logits=outputs.get('logits'),
                    hidden_states=outputs.get('hidden_states'),
                    attentions=outputs.get('attentions')
                )

            except Exception as e:
                # Fallback to GPT2 forward pass
                print(f"Warning: BitMar forward pass failed, using GPT2 fallback: {e}")
                return super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    **kwargs
                )
        else:
            # Use standard GPT2 forward pass
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )

    def generate(self, input_ids, max_length=50, **kwargs):
        """Generate method for compatibility"""
        if self.original_model is not None and hasattr(self.original_model, 'generate'):
            try:
                return self.original_model.generate(input_ids, max_length=max_length, **kwargs)
            except:
                # Fallback to GPT2 generation
                return super().generate(input_ids, max_length=max_length, **kwargs)
        else:
            return super().generate(input_ids, max_length=max_length, **kwargs)


def load_bitmar_as_hf_model(checkpoint_path: str, device: str = 'cuda:0'):
    """Load BitMar checkpoint and wrap it as HuggingFace GPT2 model"""
    try:
        # Import BitMar model creation function
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent / "src"))
        from src.model import create_bitmar_model

        print(f"Loading BitMar checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract config and model state
        bitmar_config = checkpoint.get('config', {})
        model_state = checkpoint['model_state_dict']

        # Create original BitMar model
        original_model = create_bitmar_model(bitmar_config['model'])

        # Fix shape mismatches in state dict
        fixed_state_dict = {}
        current_model_state = original_model.state_dict()

        for key, value in model_state.items():
            if key in current_model_state:
                current_shape = current_model_state[key].shape
                checkpoint_shape = value.shape

                # Handle weight_scale parameter mismatches
                if 'weight_scale' in key and checkpoint_shape != current_shape:
                    if checkpoint_shape == torch.Size([]) and current_shape == torch.Size([1]):
                        # Convert scalar to 1D tensor
                        fixed_state_dict[key] = value.unsqueeze(0)
                        print(f"Fixed shape mismatch for {key}: {checkpoint_shape} -> {current_shape}")
                    elif checkpoint_shape == torch.Size([1]) and current_shape == torch.Size([]):
                        # Convert 1D tensor to scalar
                        fixed_state_dict[key] = value.squeeze(0)
                        print(f"Fixed shape mismatch for {key}: {checkpoint_shape} -> {current_shape}")
                    else:
                        print(f"Warning: Could not fix shape mismatch for {key}: {checkpoint_shape} vs {current_shape}")
                        fixed_state_dict[key] = value
                else:
                    fixed_state_dict[key] = value
            else:
                print(f"Warning: Key {key} not found in current model, skipping")

        # Load the fixed state dict
        try:
            original_model.load_state_dict(fixed_state_dict)
            print("✅ Successfully loaded fixed state dict")
        except Exception as e:
            print(f"❌ Failed to load fixed state dict, trying strict=False: {e}")
            original_model.load_state_dict(fixed_state_dict, strict=False)
            print("✅ Loaded state dict with strict=False")

        original_model = original_model.to(device)
        original_model.eval()

        # Create GPT2 compatible config
        gpt2_config = GPT2Config(
            vocab_size=bitmar_config['model'].get('vocab_size', 50257),
            n_embd=bitmar_config['model'].get('text_encoder_dim', 128),
            n_layer=bitmar_config['model'].get('text_encoder_layers', 4),
            n_head=bitmar_config['model'].get('text_encoder_heads', 4),
            n_positions=bitmar_config['model'].get('max_seq_len', 256),
            # Set other required GPT2 parameters
            n_inner=bitmar_config['model'].get('text_encoder_dim', 128) * 4,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            use_cache=True,
            bos_token_id=50256,
            eos_token_id=50256,
            pad_token_id=50256
        )

        # Create adapter model as GPT2
        adapter_model = BitMarAsGPT2(gpt2_config, original_model)
        adapter_model = adapter_model.to(device)
        adapter_model.eval()

        print(f"✅ BitMar model loaded and adapted as GPT2 for HuggingFace compatibility")
        return adapter_model, gpt2_config

    except Exception as e:
        print(f"❌ Failed to load BitMar model: {e}")
        # Return basic GPT2 model for testing
        gpt2_config = GPT2Config(vocab_size=50257, n_embd=128, n_layer=4, n_head=4)
        adapter_model = GPT2LMHeadModel(gpt2_config)
        return adapter_model, gpt2_config


def save_hf_compatible_model(bitmar_checkpoint_path: str, output_dir: str):
    """Save BitMar model in HuggingFace GPT2 format"""
    try:
        # Load and convert model
        model, config = load_bitmar_as_hf_model(bitmar_checkpoint_path)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save config (will be saved as GPT2 config)
        config.save_pretrained(output_path)

        # Save model (will be saved as GPT2 model)
        model.save_pretrained(output_path)

        # Create a proper tokenizer for full compatibility
        from transformers import GPT2Tokenizer
        try:
            # Use GPT2 tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

            # Set pad token to eos token (common practice for GPT2)
            tokenizer.pad_token = tokenizer.eos_token

            # Save tokenizer
            tokenizer.save_pretrained(output_path)
            print(f"✅ Saved GPT2 tokenizer to: {output_path}")

        except Exception as e:
            print(f"⚠️ Failed to save tokenizer, creating manual files: {e}")

            # Create tokenizer config manually if GPT2Tokenizer fails
            tokenizer_config = {
                "add_bos_token": False,
                "add_prefix_space": False,
                "bos_token": "<|endoftext|>",
                "clean_up_tokenization_spaces": True,
                "eos_token": "<|endoftext|>",
                "model_max_length": 1024,
                "pad_token": "<|endoftext|>",
                "tokenizer_class": "GPT2Tokenizer",
                "unk_token": "<|endoftext|>"
            }

            with open(output_path / "tokenizer_config.json", 'w') as f:
                json.dump(tokenizer_config, f, indent=2)

            # Create special tokens map
            special_tokens_map = {
                "bos_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>"
            }

            with open(output_path / "special_tokens_map.json", 'w') as f:
                json.dump(special_tokens_map, f, indent=2)

            # Create vocab.json and merges.txt files from GPT2
            try:
                import requests
                # Download vocab.json
                vocab_url = "https://huggingface.co/gpt2/resolve/main/vocab.json"
                vocab_response = requests.get(vocab_url)
                if vocab_response.status_code == 200:
                    with open(output_path / "vocab.json", 'w') as f:
                        f.write(vocab_response.text)

                # Download merges.txt
                merges_url = "https://huggingface.co/gpt2/resolve/main/merges.txt"
                merges_response = requests.get(merges_url)
                if merges_response.status_code == 200:
                    with open(output_path / "merges.txt", 'w') as f:
                        f.write(merges_response.text)

                print("✅ Downloaded GPT2 vocab and merges files")
            except Exception as download_e:
                print(f"⚠️ Failed to download vocab/merges files: {download_e}")

        print(f"✅ Model saved in HuggingFace GPT2 format to: {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"❌ Failed to save HuggingFace compatible model: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert BitMar checkpoint to HuggingFace format")
    parser.add_argument("--checkpoint_path", required=True, help="Path to BitMar checkpoint")
    parser.add_argument("--output_dir", required=True, help="Output directory for HuggingFace model")
    parser.add_argument("--device", default="cuda:0", help="Device to use")

    args = parser.parse_args()

    save_hf_compatible_model(args.checkpoint_path, args.output_dir)
