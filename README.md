# ai-era-assignment11
Assignment 11 Tokenizer - Sanskrit Tokenizer

---
title: Sanskrit BPE Tokenizer
emoji: 📚
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
---

# Sanskrit BPE Tokenizer

This is a Byte-Pair Encoding (BPE) tokenizer specifically designed for Sanskrit text. It provides a web interface for both training the tokenizer and using it to encode/decode text.

## Features

- Train BPE tokenizer on custom Sanskrit text
- Tokenize Sanskrit text using the trained model
- Verify tokenization accuracy through decode/encode cycle
- User-friendly web interface

## Usage

1. Go to the "Train" tab and paste your Sanskrit training text
2. Click "Train Tokenizer" to train the model
3. Switch to the "Tokenize" tab to tokenize new text
4. Enter text and click "Tokenize" to see the results

## Example Text
```
चीराण्यपास्याज्जनकस्य कन्या नेयं प्रतिज्ञा मम दत्तपूर्वा।
यथासुखं गच्छतु राजपुत्री वनं समग्रा सह सर्वरत्नैः॥
```

# To deploy this to Hugging Face Spaces:

1. Create a new Space on Hugging Face:
```bash
huggingface-cli login
huggingface-cli repo create sanskrit-tokenizer-demo --type space
```
2. Initialize git and push your code:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://huggingface.co/spaces/your-username/sanskrit-tokenizer-demo
git push -u origin main
```

# Steps to Run Locally
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the requirements and the Hugging Face CLI:
   ```bash
    pip install -r requirements.txt
    pip install --upgrade huggingface-hub
   ```

4. To run the app:
   ```bash
   python src/app.py
   ```
   The interface will be available at `http://localhost:7860` by default.

# Logs

Orignal (before Hugging Space code):
```
(venv) gitesh.grover@Giteshs-MacBook-Pro ai-era-assignment11 % python src/main.py
Reading file...
Length of training dataset for token 620958
UTF-8 tokens length (without considering regex) 1719701
Preprocessing tokens...
UTF-8 Split tokens length (due to regex) 77799
Training BPE...
Learning BPE: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 4744/4744 [06:21<00:00, 12.45it/s]
Tokenizer training completed in 381.08 seconds
Vocab size: 5000
Original tokens length 1719701, while updated tokens length 77799
Compression Ratio 22.10
Saving Tokenizer Vocab in files...
Testing the validity of tokenizer...
True
```

After Hugging Space implementation:
```
Learning BPE: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 4744/4744 [03:25<00:00, 23.07it/s]

(Huggingface app output)
Training completed! Vocabulary size: 5000 and Compression Ratio: 9.42

```