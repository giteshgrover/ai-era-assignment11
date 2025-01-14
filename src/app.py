import gradio as gr
from tokenizer import sanskrit_token_preprocessor, bpeAlgo, create_vocab, save_paired_tokens_vocab, save_vocab, encode, decode

# Created a `TokenizerModel` class to maintain state between calls
# - Added a web interface with separate tabs for training and tokenization
# - Included example texts and documentation
# - Made the interface more user-friendly with clear labels and instructions
# - Added JSON output for better visualization of results

# The app provides two main functionalities:
# 1. Training the tokenizer on custom text
# 2. Tokenizing new text using the trained model
class TokenizerModel:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.paired_tokens_vocab = None
        self.vocab = None
    
    def train(self, text):
        # Preprocess tokens
        tokens = sanskrit_token_preprocessor(text)
        tokens = [list(tok.encode('utf-8')) for tok in tokens]
        orig_tokens_len = len([tok for toks in tokens for tok in toks])
        print(type(orig_tokens_len))
        
        # Train BPE
        try:
            self.paired_tokens_vocab, encoded_tokens = bpeAlgo(tokens, self.vocab_size - 256, 256)
            self.vocab = create_vocab(self.paired_tokens_vocab)
        except ValueError as e:
             # Display the error message in your UI
            return f"Error during training: {str(e)}"
        
        save_paired_tokens_vocab("data/paired_tokens.bpe", self.paired_tokens_vocab)
        save_vocab( "data/vocab.bpe", self.vocab)

        # Calculate compression ratio
        compression_ratio = orig_tokens_len / len(encoded_tokens)
        
        return f"Training completed! Vocabulary size: {len(self.vocab)} and Compression Ratio: {compression_ratio:.2f}"
    
    def tokenize(self, text):
        if self.paired_tokens_vocab is None:
            return "Please train the tokenizer first!"
        
        encoded = encode(text, self.paired_tokens_vocab)
        decoded = decode(encoded, self.vocab)
        
        return {
            "Encoded tokens": encoded,
            "Decoded text": decoded,
            "Matches original": decoded == text
        }

# Create global tokenizer instance
tokenizer = TokenizerModel()

def train_tokenizer(text):
    return tokenizer.train(text)

def process_text(text):
    return tokenizer.tokenize(text)

# Create the Gradio interface
with gr.Blocks(title="Sanskrit BPE Tokenizer") as demo:
    gr.Markdown("# Sanskrit BPE Tokenizer")
    gr.Markdown("This tokenizer implements Byte-Pair Encoding (BPE) for Sanskrit text.")
    
    with gr.Tab("Train"):
        train_input = gr.Textbox(
            label="Training Text",
            placeholder="Enter Sanskrit text for training...",
            lines=5
        )
        train_button = gr.Button("Train Tokenizer")
        train_output = gr.Textbox(label="Training Result")
        
        train_button.click(
            train_tokenizer,
            inputs=train_input,
            outputs=train_output
        )
    
    with gr.Tab("Tokenize"):
        text_input = gr.Textbox(
            label="Input Text",
            placeholder="Enter Sanskrit text to tokenize...",
            lines=3
        )
        tokenize_button = gr.Button("Tokenize")
        result_output = gr.JSON(label="Results")
        
        tokenize_button.click(
            process_text,
            inputs=text_input,
            outputs=result_output
        )
    
    gr.Markdown("""
    ### Example texts:
    ```
    चीराण्यपास्याज्जनकस्य कन्या नेयं प्रतिज्ञा मम दत्तपूर्वा।
    यथासुखं गच्छतु राजपुत्री वनं समग्रा सह सर्वरत्नैः॥
    ```
    """)

if __name__ == "__main__":
    demo.launch() 