# ai-era-assignment11
Assignment 11 Tokenizer - Sanskrit Tokenizer

# Steps to Run Locally
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run tests:
   ```bash
   pytest tests/
   ```

4. Train and Create vocab of tokens:
   ```bash
   python src/main.py
   ```

# To deploy to GitHub
1. Create a new GitHub repository
2. Initialize git in your local project:
   ```bash
   git init
   ```
3. Push your code to the new repository:
   ```bash
   git remote add origin https://github.com/your-username/your-repo.git
   git branch -M main
   git add .
   git commit -m "Initial commit"
   git push -u origin main
   ```

4. The GitHub Actions workflow will automatically trigger when you push to the repository. It will:
   - Set up the Python environment
   - Install dependencies
   - Run all tests

# Logs

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