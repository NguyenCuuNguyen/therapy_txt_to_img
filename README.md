## Fine-Tuning Process:
    SDXL and Kandinsky: Use Hugging Face’s diffusers library with DreamBooth or Low-Rank Adaptation (LoRA) to fine-tune on the therapy dataset. LoRA is preferred for lower memory usage on the RTX 4090.
    Karlo: Fine-tune the unCLIP model using a similar dataset, focusing on text-image alignment.
    DeepFloyd IF: Fine-tune Stage 1 (text-to-image) and Stage 2 (super-resolution) separately, as Stage 3 (upscaler) is less critical.


## Model-Specific Adjustments::
    SDXL: Optimize for high-resolution outputs (1024x1024). Use fp16 and xFormers for memory efficiency. Fine-tune with LoRA to adapt to therapy-specific imagery.
    Kandinsky 2.2: Strong at artistic styles. Adjust negative prompts to avoid irrelevant elements (e.g., “text, logos, blurry”).
    Karlo: Good for creative outputs but less flexible. Fine-tune text embeddings to better align with therapy topics.
    DeepFloyd IF: Excels at detailed textures but struggles with long prompts. Use a prompt summarizer and fine-tune Stage 1 for therapy-specific content.


## Evaluation Metrics:
    Qualitative Scores: Use the two raters’ scores (from the original server) to compare models. Define a scoring rubric (e.g., 1–5 scale for relevance, emotional impact, clarity).
    Automated Metrics: Use CLIP-based similarity scores to measure alignment between generated images and input prompts.
    Human Evaluation: Conduct a side-by-side comparison of model outputs for a subset of prompts to identify the best performer.


Run All Models:
Use run_custom_experiment.py to generate images for all models on the same prompts.
Copy images to /mnt/disk4t/psyc/outputs/images/custom_model_exp/ for evaluation:
bash

```scp -r experiments/*/images/* user@original_server:/mnt/disk4t/psyc/outputs/images/custom_model_exp/```


Run Experiments:
```python scripts/run_custom_experiment.py```


Copy images for evaluation:
```scp -r experiments/*/images/* user@original_server:/mnt/disk4t/psyc/outputs/images/custom_model_exp/```


Use fp16 precision and use_safetensors for all models to reduce VRAM usage.
Enable xFormers for SDXL and Kandinsky to optimize attention mechanisms.
Use model offloading (enable_model_cpu_offload) for DeepFloyd IF to move unused model parts to CPU.


```rm -rf ~/.cache/huggingface/hub/models--*```

```python scripts/run_finetune_multiple_epochs.py```

## Iteration 1: Generate images
Run ```python scripts/generate_images_from_finetuned.py``` to
1. Read prompts and specificity details from a CSV file (you'll need to create this based on your example).
2. Identify the saved "best" and "last" LoRA checkpoints for each model configuration you trained.
3. Load the appropriate base model (SDXL, Kandinsky).
4. Load the corresponding LoRA weights onto the base model.
5. Generate an image for each prompt using the fine-tuned model.
6. Save the generated images with informative names.



## Iterate 2: 
Changes: + Summarize prompt instead of truncated by CLIP tokenizer
         + Add refiner

SDXL:
    cross validation: python scripts/cross_validation_sdxl_base_refiner.py 
    train with best result: python scripts/train_sdxl_base_refiner.py


#how to run:
cd generated_images
mkdir -p /home/iris/Documents/deep_learning/experiments/{new_finetune}/{sdxl,kandinsky,karlo,deepfloyd_if}
export PYTHONPATH="/home/iris/Documents/deep_learning"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

iter 1:
    finetune: sdxl, kandinsky
    new_finetune: deepfloyd

iter 2: 
    bigger_kadinsky

iter 3:
    



## Iteration 0: New finetune model focuses more on Karlo and DeepFloydIF
Clear hugging face cache: rm -rf ~/.cache/huggingface/hub/models--*


## DIRECTORY:
Setup to train originally with COCO: includes sdxl, kandinsky, karlo and deepfloyd but only sdxl and kandinsky really works. Prompt is truncated by CLIP.
    finetune_model.py
    run_finetune_multiple_epochs.py
    generate_images_from_finetuned.py


New set up emphasized on karlo and deepfloyd but that does not really work because deepfloyd has OOM error and karlo also has error, could not train sucessfully