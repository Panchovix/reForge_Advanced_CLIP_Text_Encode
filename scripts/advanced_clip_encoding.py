import logging
import gradio as gr
from modules import scripts
from advclip.adv_encode import advanced_encode, advanced_encode_XL
import torch

def safe_encode_token_weights(model, token_weight_pairs):
    tokens = [t for t, _ in token_weight_pairs]
    weights = [w for _, w in token_weight_pairs]
    
    # Convert tokens to tensor
    tokens_tensor = model.tokenizer.convert_tokens_to_ids(tokens)
    tokens_tensor = torch.tensor([tokens_tensor], device=model.device)
    
    # Apply weights
    weights_tensor = torch.tensor([weights], device=model.device, dtype=torch.float32)
    
    # Encode
    with torch.no_grad():
        try:
            # Try with intermediate_output first
            outputs = model.transformer(tokens_tensor, output_hidden_states=True, intermediate_output=model.layer_idx)
            hidden_states = outputs.hidden_states[model.layer_idx]
        except TypeError:
            # If that fails, try without intermediate_output
            outputs = model.transformer(tokens_tensor, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Use the last layer
    
    # Apply weights to hidden states
    weighted_hidden_states = hidden_states * weights_tensor.unsqueeze(-1)
    
    # Get pooled output
    pooled_output = model.transformer.pooler(hidden_states)
    
    return weighted_hidden_states, pooled_output

class AdvancedCLIPTextEncodeScript(scripts.Script):
    def __init__(self):
        self.enabled = False
        self.token_normalization = "none"
        self.weight_interpretation = "comfy"
        self.affect_pooled = "disable"
        self.balance = 0.5
        self.use_sdxl = False
        self.text = ""
        self.text_g = ""

    sorting_priority = 17

    def title(self):
        return "Advanced CLIP Text Encode for reForge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML("<p><i>Adjust the settings for Advanced CLIP Text Encode.</i></p>")

            enabled = gr.Checkbox(label="Enable Advanced CLIP Text Encode", value=self.enabled)

            use_sdxl = gr.Checkbox(label="Use SDXL", value=self.use_sdxl)

            with gr.Group() as sd_group:
                text = gr.Textbox(label="Text", lines=3, value=self.text)

            with gr.Group() as sdxl_group:
                text_l = gr.Textbox(label="Text (L)", lines=3, value=self.text)
                text_g = gr.Textbox(label="Text (G)", lines=3, value=self.text_g)
                balance = gr.Slider(label="Balance", minimum=0.0, maximum=1.0, step=0.01, value=self.balance)

            token_normalization = gr.Radio(
                ["none", "mean", "length", "length+mean"],
                label="Token Normalization",
                value=self.token_normalization
            )

            weight_interpretation = gr.Radio(
                ["comfy", "A1111", "compel", "comfy++", "down_weight"],
                label="Weight Interpretation",
                value=self.weight_interpretation
            )

            affect_pooled = gr.Radio(
                ["disable", "enable"],
                label="Affect Pooled",
                value=self.affect_pooled
            )

            def update_visibility(use_sdxl):
                return (
                    gr.Group.update(visible=not use_sdxl),
                    gr.Group.update(visible=use_sdxl)
                )

            use_sdxl.change(
                update_visibility,
                inputs=[use_sdxl],
                outputs=[sd_group, sdxl_group]
            )

        return (enabled, use_sdxl, text, text_l, text_g, token_normalization, weight_interpretation, affect_pooled, balance)

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 9:
            (self.enabled, self.use_sdxl, self.text, self.text_l, self.text_g,
             self.token_normalization, self.weight_interpretation, self.affect_pooled, self.balance) = args[:9]
        else:
            logging.warning("Not enough arguments provided to process_before_every_sampling")
            return

        if not self.enabled:
            return

        clip = p.sd_model.forge_objects.clip

        # Override the encode_token_weights function
        clip.cond_stage_model.encode_token_weights = lambda x: safe_encode_token_weights(clip.cond_stage_model, x)

        if self.use_sdxl:
            embeddings_final, pooled = advanced_encode_XL(
                clip, self.text_l, self.text_g, self.token_normalization,
                self.weight_interpretation, w_max=1.0, clip_balance=self.balance,
                apply_to_pooled=self.affect_pooled == "enable"
            )
        else:
            embeddings_final, pooled = advanced_encode(
                clip, self.text, self.token_normalization, self.weight_interpretation,
                w_max=1.0, apply_to_pooled=self.affect_pooled == "enable"
            )

        # Replace the original conditioning with the new advanced encoding
        c = [[embeddings_final, {"pooled_output": pooled}]]
        uc = clip.encode_from_tokens(clip.tokenize(""))

        p.batch_conds = [c] * p.batch_size
        p.batch_negative_conds = [uc] * p.batch_size

        # Clear the original prompts as we're using advanced encoding
        p.prompt = ""
        p.negative_prompt = ""
        p.all_prompts = [p.prompt] * p.batch_size
        p.all_negative_prompts = [p.negative_prompt] * p.batch_size

        p.extra_generation_params.update({
            "Advanced CLIP Encode": self.enabled,
            "Token Normalization": self.token_normalization,
            "Weight Interpretation": self.weight_interpretation,
            "Affect Pooled": self.affect_pooled,
            "SDXL Mode": self.use_sdxl,
            "SDXL Balance": self.balance if self.use_sdxl else None,
        })

        logging.debug(f"Advanced CLIP Encode: Enabled: {self.enabled}, SDXL Mode: {self.use_sdxl}")

        return