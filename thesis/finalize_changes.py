import sys
import re
import os

def process_latex_changes(text):
    has_changes = False
    
    while True:
        # Match \added, \deleted, \replaced, optionally with [...] arguments, followed by {
        match = re.search(r'\\(added|deleted|replaced)\s*(?:\[[^\]]*\])?\s*\{', text)
        if not match:
            break
            
        cmd = match.group(1)
        start_idx = match.start()
        
        # content_start is the position of '{'
        content_start = match.end() - 1 
        assert text[content_start] == '{'
        
        brace_level = 0
        content_end = -1
        for i in range(content_start, len(text)):
            if text[i] == '{':
                brace_level += 1
            elif text[i] == '}':
                brace_level -= 1
                if brace_level == 0:
                    content_end = i
                    break
        
        if content_end == -1:
            print(f"Warning: Unbalanced braces found for \\{cmd}")
            break
            
        block1 = text[content_start+1:content_end]
        
        if cmd == 'added':
            # Remove \added{ and } but keep the content block1
            text = text[:start_idx] + block1 + text[content_end+1:]
            has_changes = True
        elif cmd == 'deleted':
            # Remove the whole \deleted{...} command
            text = text[:start_idx] + text[content_end+1:]
            has_changes = True
        elif cmd == 'replaced':
            # Find the second block {old} for \replaced{new}{old}
            after_first_block = content_end + 1
            second_block_start = text.find('{', after_first_block)
            
            if second_block_start != -1 and text[after_first_block:second_block_start].strip() == '':
                brace_level = 0
                second_block_end = -1
                for i in range(second_block_start, len(text)):
                    if text[i] == '{':
                        brace_level += 1
                    elif text[i] == '}':
                        brace_level -= 1
                        if brace_level == 0:
                            second_block_end = i
                            break
                if second_block_end != -1:
                    # For \replaced, keep block1 (new text), discard block2 (old text)
                    text = text[:start_idx] + block1 + text[second_block_end+1:]
                    has_changes = True
                else:
                    print(f"Warning: Unbalanced second block for \\replaced")
                    break
            else:
                print(f"Warning: Missing second block for \\replaced")
                break
                
    return text, has_changes

if __name__ == "__main__":
    thesis_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Scanning directory: {thesis_dir}")
    print("Looking for .tex files...")
    
    updated_files = 0
    scanned_files = 0
    
    for root, dirs, files in os.walk(thesis_dir):
        # Skip git or backup directories to be safe
        if '.git' in root or 'backup' in root.lower():
            continue
            
        for file in files:
            if file.endswith('.tex') and not file.endswith('.bak'):
                filepath = os.path.join(root, file)
                scanned_files += 1
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                new_content, changed = process_latex_changes(content)
                
                if changed:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Updated: {os.path.relpath(filepath, thesis_dir)}")
                    updated_files += 1
                    
    print(f"\nScan complete. Scanned {scanned_files} active .tex files.")
    print(f"Successfully finalized changes in {updated_files} files.")
