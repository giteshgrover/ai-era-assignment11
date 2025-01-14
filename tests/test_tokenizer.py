import pytest
from tokenizer import sanskrit_token_preprocessor, bpeAlgo, create_vocab, encode, decode

@pytest.fixture
def sample_text():
    return """चीराण्यपास्याज्जनकस्य कन्या नेयं प्रतिज्ञा मम दत्तपूर्वा।
अजीवनाहेण मया नृशंसा कृता प्रतिज्ञा नियमेन तावत्। I am going"""

@pytest.fixture
def small_vocab_size():
    return 300  # Small vocab size for testing

def test_sanskrit_token_preprocessor():
    text = "Hello! नमस्ते 123"
    tokens = sanskrit_token_preprocessor(text)
    assert isinstance(tokens, list)
    assert all(isinstance(token, str) for token in tokens)

def test_encode_decode_roundtrip(sample_text, small_vocab_size):
    # Prepare tokens
    tokens = sanskrit_token_preprocessor(sample_text)
    tokens = [list(tok.encode('utf-8')) for tok in tokens]
    
    # Train BPE
    paired_tokens_vocab, _ = bpeAlgo(tokens, small_vocab_size - 256, 256)
    vocab = create_vocab(paired_tokens_vocab)
    
    # Test encode-decode roundtrip
    encoded = encode(sample_text, paired_tokens_vocab)
    decoded = decode(encoded, vocab)
    assert decoded == sample_text

def test_vocab_size_limit(sample_text, small_vocab_size):
    tokens = sanskrit_token_preprocessor(sample_text)
    tokens = [list(tok.encode('utf-8')) for tok in tokens]
    
    paired_tokens_vocab, _ = bpeAlgo(tokens, small_vocab_size - 256, 256)
    vocab = create_vocab(paired_tokens_vocab)
    
    # Check if vocab size is within limits
    assert len(vocab) <= small_vocab_size

def test_empty_input():
    empty_text = ""
    tokens = sanskrit_token_preprocessor(empty_text)
    assert tokens == []
    
    # Test encode/decode with empty input
    paired_tokens_vocab = {}
    vocab = create_vocab(paired_tokens_vocab)
    encoded = encode(empty_text, paired_tokens_vocab)
    decoded = decode(encoded, vocab)
    assert decoded == empty_text

def test_special_characters():
    special_text = "!@#$%^&*()_+ १२३"
    tokens = sanskrit_token_preprocessor(special_text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0 