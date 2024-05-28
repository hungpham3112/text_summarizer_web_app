def split_into_chunks(text, max_tokens=1024):
    words = text.split()
    chunks, current_chunk, current_length = [], [], 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # Add 1 for the space
        if current_length >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [], 0
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
