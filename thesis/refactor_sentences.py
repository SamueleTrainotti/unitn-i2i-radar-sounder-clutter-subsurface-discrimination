import re
import os

def protect_nested_blocks(text, command_name, protected_blocks):
    """
    Finds all instances of \command_name{...} and protects them,
    correctly handling nested braces.
    """
    search_str = f"\\{command_name}{{"
    start_idx = 0
    
    while True:
        idx = text.find(search_str, start_idx)
        if idx == -1:
            break
            
        # Found the start of the command. Now find the matching closing brace.
        brace_count = 1
        pos = idx + len(search_str)
        
        while pos < len(text) and brace_count > 0:
            if text[pos] == '{':
                brace_count += 1
            elif text[pos] == '}':
                brace_count -= 1
            pos += 1
            
        if brace_count == 0:
            # We found the full block
            block = text[idx:pos]
            protected_blocks.append(block)
            placeholder = f"__PROTECTED_BLOCK_{len(protected_blocks)-1}__"
            text = text[:idx] + placeholder + text[pos:]
            start_idx = idx + len(placeholder)
        else:
            # Unbalanced braces? Just move on to avoid infinite loop
            start_idx = idx + len(search_str)
            
    return text

def split_latex_sentences(text):
    protected_blocks = []
    
    def protect(match):
        protected_blocks.append(match.group(0))
        return f"__PROTECTED_BLOCK_{len(protected_blocks)-1}__"

    # 1. Protect Comments
    # We must be careful not to protect the newline itself, just the comment up to the newline.
    text = re.sub(r'%.*?$', protect, text, flags=re.MULTILINE)

    # 2. Protect math blocks
    text = re.sub(r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}', protect, text, flags=re.DOTALL)
    text = re.sub(r'\$\$.*?\$\$', protect, text, flags=re.DOTALL)
    text = re.sub(r'\\\[.*?\\\]', protect, text, flags=re.DOTALL)
    
    # 3. Protect Inline math
    text = re.sub(r'\$(?:[^$]|\\\$)+\$', protect, text)
    
    # 4. Protect fragile LaTeX commands that shouldn't be broken or can cause runaway arguments
    # \caption{...}, \footnote{...}, \section{...}, \subsection{...}, \chapter{...}
    for cmd in ['caption', 'footnote', 'section', 'subsection', 'subsubsection', 'chapter', 'paragraph', 'textbf', 'textit', 'enquote']:
        text = protect_nested_blocks(text, cmd, protected_blocks)
    
    # Negative lookbehinds for abbreviations
    abbrevs = r"(?<!e\.g)(?<!i\.e)(?<!et al)(?<!Fig)(?<!Eq)(?<!Ref)(?<!al)(?<!vs)(?<!approx)(?<!cf)(?<!Dr)(?<!Prof)(?<!Mr)(?<!Ms)(?<!Sec)(?<!Chap)(?<!\b[A-Z])"
    
    # The ending marker: punctuation, optional quotes/parens, optional citation.
    sentence_end = r'(' + abbrevs + r'(?:[\.\!\?]+(?:[\'\"\)\]\}]+)?(?:(?:~|\\ )?(?:\\cite|\\ref)\{[^}]+\})?))'
    spaces = r'([ \t]+)'
    # Next start: Capital letter or common start macro
    next_start = r'(?=[A-Z]|\\(?:textbf|textit|emph|enquote|replaced|added))'
    
    pattern = re.compile(sentence_end + spaces + next_start)
    
    # Replace spaces with a single newline
    text = pattern.sub(r'\1\n', text)
    
    # Restore protected blocks
    for i in range(len(protected_blocks)-1, -1, -1):
        text = text.replace(f"__PROTECTED_BLOCK_{i}__", protected_blocks[i])
        
    return text

if __name__ == "__main__":
    thesis_dir = os.path.dirname(os.path.abspath(__file__))
    
    updated_files = 0
    scanned_files = 0
    
    for root, dirs, files in os.walk(thesis_dir):
        if '.git' in root or 'backup' in root.lower() or 'images' in root.lower():
            continue
            
        for file in files:
            if file.endswith('.tex') and not file.endswith('.bak'):
                filepath = os.path.join(root, file)
                scanned_files += 1
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                new_content = split_latex_sentences(content)
                
                if new_content != content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    updated_files += 1
                    
    print(f"Successfully refactored {updated_files} files out of {scanned_files} scanned.")
