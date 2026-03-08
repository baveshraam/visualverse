# Story Generator Model Comparison: 5K vs 10K+ Training

## Overview
Two versions of story generation models have been trained with different dataset sizes to allow quality comparison.

## Dataset Specifications

### 5K Version (Original)
- **English**: 5,000 samples (WritingPrompts dataset)
- **Hindi**: 2,000 samples (synthetic templates)
- **Tamil**: 2,000 samples (synthetic templates)
- **Total**: 9,000 samples
- **Training Time**: ~25 minutes on RTX 3050 Ti

### 10K+ Version (Enhanced)
- **English**: 15,000 samples (WritingPrompts + synthetic)
- **Hindi**: 5,000 samples (synthetic templates with more variety)
- **Tamil**: 5,000 samples (synthetic templates with more variety)
- **Total**: 25,000 samples (2.7x increase)
- **Training Time**: ~60-80 minutes on RTX 3050 Ti (estimated)

## GPU Optimization Settings Applied

Based on OOM errors encountered during initial training, the following optimizations were applied:

### Memory-Critical Settings
```python
batch_size = 1                      # Reduced from 4
max_length = 256                    # Reduced from 512
gradient_accumulation_steps = 8     # Increased from 4
fp16 = False                        # Disabled mixed precision
```

### Additional Optimizations
- **CUDA Memory Fragmentation**: `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64`
- **Garbage Collection**: Explicit GPU memory cleanup between language trainings
- **Gradient Clipping**: `max_grad_norm=1.0` for training stability

## Model Storage Locations

### 5K Models
```
backend/models/story_model_en/
backend/models/story_model_hi/
backend/models/story_model_ta/
```

### 10K+ Models
```
backend/models/story_model_en_10k/
backend/models/story_model_hi_10k/
backend/models/story_model_ta_10k/
```

## Training Results Comparison

### 5K Version Results
| Language | Training Timeᵃ | Final Loss | Perplexity | Samples |
|----------|---------------|------------|------------|---------|
| English  | 14.3 min      | 2.2233     | 9.24       | 5,000   |
| Hindi    | 5.8 min       | 0.0028     | 1.00       | 2,000   |
| Tamil    | 5.8 min       | 0.0029     | 1.00       | 2,000   |

ᵃ On NVIDIA RTX 3050 Ti Laptop GPU (4GB VRAM)

### 10K+ Version Results
*Will be updated after training completes*

| Language | Training Timeᵃ | Final Loss | Perplexity | Samples |
|----------|---------------|------------|------------|---------|
| English  | TBD           | TBD        | TBD        | 15,000  |
| Hindi    | TBD           | TBD        | TBD        | 5,000   |
| Tamil    | TBD           | TBD        | TBD        | 5,000   |

## Expected Quality Improvements

### 10K+ Version Benefits
1. **More Diverse Outputs**: 2.7x more training data = more varied story patterns
2. **Better Genre Coverage**: English model sees 15K prompts across all genres
3. **Improved Coherence**: More examples help model learn better narrative structures
4. **Reduced Repetition**: Larger dataset reduces overfitting and repetitive patterns
5. **Better Keyword Understanding**: More examples of keyword-to-story mappings

### Potential Trade-offs
- **Longer Training Time**: ~3x longer training (25 min → 75 min)
- **Slightly Larger Model Files**: More diverse vocabulary may increase size marginally
- **Same Inference Speed**: Generation speed remains identical

## How to Switch Between Models

### Using 5K Models (Default)
```python
# In backend/story_gen/story_generator.py
model_path = self.models_dir / f"story_model_{language}"
```

### Using 10K+ Models
```python
# In backend/story_gen/story_generator.py
model_path = self.models_dir / f"story_model_{language}_10k"
```

## Testing & Comparison Methodology

### Suggested Comparison Tests

1. **Same Keywords Test**
   - Input: `genre: sci-fi, hero: astronaut, setting: Mars, plot: survival`
   - Generate with both 5K and 10K models
   - Compare story quality, coherence, creativity

2. **Genre Diversity Test**
   - Test keywords from different genres: fantasy, mystery, romance, horror
   - Check if 10K model shows better genre understanding

3. **Keyword Complexity Test**
   - Simple: `hero, journey, magic`
   - Complex: `genre: dystopian, protagonist: rebel, setting: underground city, themes: freedom and resistance`
   - Compare handling of complex vs simple prompts

4. **Multilingual Quality Test**
   - Test Hindi and Tamil generation with cultural elements
   - Check grammatical correctness and narrative flow

### Metrics to Compare
- **Story Length**: Word count consistency
- **Coherence**: Logical flow and narrative structure
- **Creativity**: Uniqueness and originality
- **Keyword Relevance**: How well story matches keywords
- **Language Quality**: Grammar, vocabulary richness

## Files Created

### Dataset Preparation
- `training/story_training/prepare_dataset_10k.py` - Generates 25K sample dataset
- `training/story_training/data_10k/` - Directory containing large datasets

### Training Scripts
- `training/story_training/train_story_generator_10k.py` - Training script for 10K models
- `training/train_story_models_10k.sh` - Automated training pipeline

### Documentation
- `training/MODEL_COMPARISON.md` - This file

## Running the Training

### Activate Virtual Environment
```bash
source /home/kavin/career/env/bin/activate
```

### Navigate to Training Directory
```bash
cd /home/kavin/career/college/sem6/nlp/project/nlp-main/training
```

### Run 10K+ Training Pipeline
```bash
./train_story_models_10k.sh
```

The script will:
1. Check GPU availability
2. Prepare 25,000 training samples
3. Train English model (15K samples, ~30-40 min)
4. Clear GPU memory
5. Train Hindi model (5K samples, ~15-20 min)
6. Clear GPU memory
7. Train Tamil model (5K samples, ~15-20 min)
8. Save all models to `*_10k` directories

## Troubleshooting

### If OOM Errors Occur
The script already applies all known optimizations. If you still get OOM:
1. Close all other applications
2. Train languages individually:
   ```bash
   python3 training/story_training/train_story_generator_10k.py --language en --epochs 3 --batch-size 1
   ```
3. Consider training on CPU (very slow but guaranteed):
   ```bash
   CUDA_VISIBLE_DEVICES="" python3 training/story_training/train_story_generator_10k.py --language en
   ```

### Monitoring Training
- Watch GPU memory: `nvidia-smi` in another terminal
- Training progress shown with progress bars
- Loss values printed every 50 steps

## Next Steps After Training

1. **Update Backend** to use 10K models by default
2. **A/B Testing** - Generate stories with both versions
3. **User Feedback** - Collect which version produces better stories
4. **Documentation** - Update training results in this file
5. **Model Selection** - Choose which version to deploy based on quality

---

**Note**: Keep both model versions for comparison. The 5K models serve as a good baseline and are faster to retrain if needed.
