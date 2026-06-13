# EXL3 Model Search Results

## Status: NOT FOUND

Searched for Qwen3.5-27B EXL3 models on HuggingFace:
- LoneStriker: Has EXL2 models for other Qwen variants, but no Qwen3.5-27B
- turboderp: No public Qwen3.5-27B EXL2/EXL3
- bartowski: Only has GGUF quantizations

## Screenshot Analysis

The screenshot shows EXL3 formats with specific bpw values:
- EXL3 2.10 bpw: 6.68 GB
- EXL3 3.01 bpw: 9.44 GB  
- EXL3 3.10 bpw: 9.68 GB
- EXL3 4.01 bpw: 12.27 GB
- EXL3 5.01 bpw: 15.10 GB

**Possible sources**:
1. Private/unreleased quantizations
2. Self-quantized by screenshot creator
3. Different model repo not found in search

## Alternative: Create Our Own EXL2 Quantization

If we want to compare with EXL2/EXL3:

### Option 1: Use existing EXL2 tools
```bash
# Install exllamav2
source ~/exllama_env/bin/activate
pip install exllamav2

# Quantize Qwen3.5-27B to EXL2 3.0 bpw
python -m exllamav2.convert \
    --input /path/to/Qwen3.5-27B \
    --output /home/awee/models/Qwen3.5-27B-EXL2-3.0bpw \
    --bits 3.0 \
    --calibration wikitext
```

### Option 2: Compare with available formats

We already have comprehensive GGUF comparisons:
- UD-Q2_K_XL: 9.76 GB, PPL 7.53
- TQ3_4S: 12.9 GB, PPL 6.77
- IQ4_XS: 13.27 GB, PPL 6.83

**TQ3_4S is already competitive** with similar-sized formats.

## Recommendation

**Skip EXL3 comparison** because:
1. Models not publicly available
2. Cannot reproduce screenshot test conditions
3. TQ3_4S already validated against GGUF formats
4. Creating our own EXL2 quant would take hours

**Focus instead on**:
1. Document TQ3_4S performance (315 tok/s, PPL 6.77)
2. Compare with existing GGUF formats (already done)
3. Move on to next optimization target

## Conclusion

EXL3 models from screenshot are not available for download. We have sufficient comparison data from GGUF formats to validate TQ3_4S quality and performance.
