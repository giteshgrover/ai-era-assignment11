import regex as re
from tokenizer import sanskrit_token_preprocessor, bpeAlgo, create_vocab, save_paired_tokens_vocab, save_vocab, encode, decode

class Config:
    vocab_size = 5000
    pair_token_file = "data/paired_tokens.bpe"
    vocab_file = "data/vocab.bpe"



def train_and_create_vocab():
    with open('static/sanskrit.txt', 'r', encoding='utf-8') as f: 
        text = f.read()
#     text = """चीराण्यपास्याज्जनकस्य कन्या नेयं प्रतिज्ञा मम दत्तपूर्वा। यथासुखं गच्छतु राजपुत्री वनं समग्रा सह सर्वरत्नैः॥
# अजीवनाहेण मया नृशंसा कृता प्रतिज्ञा नियमेन तावत्। त्वया हि बाल्यात् प्रतिपन्नमेतत् तन्मा 123 4 5दहेद् वेणुमिवात्मपुष्पम्॥ I am going
# """
    print(f'Length of training dataset for token {len(text)}')

    origTokensLen = len(list(text.encode('utf-8')))
    print(f'UTF-8 tokens length (without considering regex) {origTokensLen}')
    
    tokens = sanskrit_token_preprocessor(text) # Split texts to list of strings as per the preProcess rules (regex)
    # print(tokens)
    tokens = [list(tok.encode('utf-8')) for tok in tokens] # Tokens is a list of list of utf-8 strings
    print(f'UTF-8 Split tokens length (due to regex) {len(tokens)}')
    # print(tokens)

    # BPE algo on list of list of utf-8 tokens
    paired_tokens_vocab, encodedTokens = bpeAlgo(tokens, Config.vocab_size - 256, 256)
    # paired_tokens_vocab is what was called 'merges' in the class
    vocab = create_vocab(paired_tokens_vocab)
    # print(paired_tokens_vocab)
    # print(vocab)
    print(f"Vocab size: {len(vocab)}")


    print(f"Original tokens length {origTokensLen}, while updated tokens length {len(encodedTokens)}")
    print(f"Compression Ratio {origTokensLen/len(encodedTokens):.2f}")

    save_paired_tokens_vocab(Config.pair_token_file, paired_tokens_vocab)
    save_vocab( Config.vocab_file, vocab)

    ## Test
    text = """चीराण्यपास्याज्जनकस्य कन्या नेयं प्रतिज्ञा मम दत्तपूर्वा। यथासुखं गच्छतु राजपुत्री वनं समग्रा सह सर्वरत्नैः॥
अजीवनाहेण मया नृशंसा कृता प्रतिज्ञा नियमेन तावत्। त्वया हि बाल्यात् प्रतिपन्नमेतत् तन्मा 123 4 5दहेद् वेणुमिवात्मपुष्पम्॥ I am going
"""
    print(decode(encode(text, paired_tokens_vocab), vocab) == text)



if __name__ == "__main__":
    train_and_create_vocab() 