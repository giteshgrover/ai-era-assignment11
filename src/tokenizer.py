import regex as re
from tqdm import tqdm

# U+090x
# ऀ ँ ऄ अ आ इ ई उ ऊ ऋ ऌ ऍ ऎ ए

# U+091x
# ऐ ऑ ऒ ओ औ क ख ग घ ङ च छ ज झ ञ ट

# U+092x
# ठ ड ढ ण त थ द ध न ऩ प फ ब भ म य

# U+093x
# र ऱ ल ळ ऴ व श ष स हऺऻ ़ ऽ ाि

# U+094x
# ी ु ू ृ ॄ ॅ ॆ े ै ॉ ॊ ो ौ ् ॎ ॏ

# U+095x
# ॐ ॑ ॓ ॕ ॖ ॗ क़ ख़ ग़ ज़ ड़ ढ़ फ़ य़

# U+096x
# ॠ ॡ ॢ । ॥ ० १ २ ३ ४ ५ ६ ७ ८ ९

# U+097x
# ॰ ॱ ॲ ॳ ॴ ॵ ॶ ॷ ॸ ॹ ॺ ॻ ॼ ॽ ॾ ॿ
def sanskrit_token_preprocessor(text):
    # pat = re.compile(r"""॥|।| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    pat = re.compile(r"""
            \ ?[\u0900-\u097F]+               # Match Sanskit words with an optional space before
            |\ ?\d+                        # Matches one or more numerical digits.
            |\ ?[^\s\u0900-\u097F\d]+    # Matches any character that is not a space, snaskrit or digit  with an optional space before 
            |\s+(?!\S)                     # trailing whitespace characters (spaces, tabs, newlines)
            |\s+                            # Match whitespace (spaces, tabs, newlines)
            |\s?[\r\n] 
        """, re.VERBOSE)
    return re.findall(pat, text)

def bpeAlgo(tokensList, total_runs, newTokenStartValue):
    # tokensList - list of list of tokens
    paired_tokens_vocab = {}
    newTokensList = list(tokensList) # copy list to keep original unchanged
    # newTokensList - list of list of tokens
    
    for i in tqdm(range(0, total_runs), desc="Learning BPE"):
        pair_stats = {}
        for tokens in newTokensList:
             pair_stats = get_pair_stats(tokens, pair_stats)
       
        # print(pair_stats)
        top_pair = max(pair_stats, key=pair_stats.get)
        newTokenVal = len(paired_tokens_vocab)+newTokenStartValue
        # Replace tokens
        newTokensList = [merge(tokens, top_pair, newTokenVal) for tokens in newTokensList]
        # print(f"replaced topPair {top_pair}'s {pair_stats.get(top_pair)} occurrences with {len(paired_tokens_vocab)+256}")
        # Add the new token to paired_tokens_vocab
        paired_tokens_vocab[top_pair] = newTokenVal

    return paired_tokens_vocab, [tok for tok in tokens for tokens in newTokensList]

# Get dictonary with key as token pairs and value as number of occurence
def get_pair_stats(toks, pair_stats): 
    for pair in zip(toks, toks[1:]):
        pair_stats[pair] = pair_stats.get(pair, 0) + 1
    
    return pair_stats

# Replaces all the occurences of pair in the tokens list with a new token
def merge(toks, pair, newTok):
    newToks = []
    i = 0
    while i < len(toks):
        if i < len(toks) - 1 and (toks[i], toks[i+1]) == pair:
            newToks.append(newTok)
            i += 2
        else:
            newToks.append(toks[i])
            i += 1
    return newToks

# vocab - key as int values from 0 - max(vocabSize) and values are bytes
def create_vocab(paired_tokens_vocab):
    vocab = {i: bytes([i]) for i in range(256)} # 0-255 bytes as is

    for (p0, p1), i in paired_tokens_vocab.items():
        vocab[i] = vocab[p0] + vocab[p1] # as we are iterating in order, we are deriving the values of later vocab from previous vocab value bytes

    return vocab

def save_paired_tokens_vocab(filepath, paired_tokens_vocab):
     with open(filepath, 'w') as f:
        f.write('Token version 1\n')
        for k,v in paired_tokens_vocab.items():
            f.write(f"{k}:{v}\n") 

def save_vocab(filepath, vocab):
     with open(filepath, 'w') as f:
        f.write('Token version 1\n')
        for k,v in vocab.items():
            f.write(f"{k}:{v}\n") 
        


##### Following methods are used to use the already created vocab ####
# paired_tokens_vocab - key as integer pairs (p0 & p1 would be between 0 & max_vocab_size) and value would be integer val between 0 - max_vocab_size
# vocab - key as int values from 0 - max(vocabSize) and values are bytes

def decode(intTokens, vocab):
    vocalTokenVals = b"".join(vocab[vocabKey] for vocabKey in intTokens) # join actual byte values of the keys from vocab
    return vocalTokenVals.decode("utf-8")  # Decode the utf-code byte sequemces back to String

def encode(text, paired_tokens_vocab):
    tokens = list(text.encode('utf-8'))

    while len(tokens) > 1: # as long as the token values can be paired. It will break within loop below
        pair_stats = get_pair_stats(tokens, {})
        # Get the matching pair from the vocab with least value. That way, if it is not found, we can stop the iteration
        pair = min(pair_stats, key=lambda k : paired_tokens_vocab.get(k, float("inf"))) # find min pair whose min value is decided based on the value found in paired_tokens_vocab
        if pair not in paired_tokens_vocab:
            return tokens
        newTok = paired_tokens_vocab[pair]
        # print(f"Replacing {pair} with {newTok}")
        tokens = merge(tokens, pair, newTok)