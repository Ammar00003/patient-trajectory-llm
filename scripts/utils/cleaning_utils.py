import re

def clean_gemma_output(text):
    # Remove ANSI escape codes - comprehensive pattern
    # Matches: \x1b[...A, [K, [13D, [1D[K, etc.
    # Pattern 1: Full escape sequences starting with \x1b
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)
    # Pattern 2: Bracket-based codes like [K, [13D, [1D
    text = re.sub(r'\[[0-9]*[A-Za-z](?:\[K)?', '', text)
    # Pattern 3: Any remaining isolated [K sequences
    text = re.sub(r'\[K', '', text)

    # Remove the ...done thinking. marker and everything before it
    marker = '...done thinking.'
    idx = text.find(marker)
    if idx != -1:
        text = text[idx + len(marker):].strip()
    else:
        # Fallback to finding start of actual output
        for keyword in ['MEDS_ON_ADMISSION', 'MEDS_ON_DISCHARGE', 'MEDICATIONS_ON_ADMISSION', 'MEDICATIONS_ON_DISCHARGE', 'EVENT_TIMELINE']:
            idx = text.find(keyword)
            if idx != -1:
                text = text[idx:].strip()
                break

    # Fix broken lines where ANSI codes split words
    # Pattern: word fragment at end of line + newline + word that completes or replaces it
    # Example: "and p\npain" -> "and pain"
    # Example: "and Spi\nSpironolactone" -> "and Spironolactone"
    lines = text.split('\n')
    merged_lines = []
    i = 0

    while i < len(lines):
        current_line = lines[i].rstrip()

        # Check if we should merge with next line
        if i + 1 < len(lines):
            next_line = lines[i + 1].lstrip()

            # Extract the last word fragment from current line
            last_word_match = re.search(r'(\S+)$', current_line)
            if last_word_match:
                last_word = last_word_match.group(1)
            else:
                last_word = ''

            # Extract the first word from next line
            first_word_match = re.search(r'^(\S+)', next_line)
            if first_word_match:
                first_word = first_word_match.group(1)
            else:
                first_word = ''

            # Check if this looks like a broken word that needs merging:
            # 1. Last word ends with a letter (potentially incomplete)
            # 2. First word starts with a letter
            # 3. Last word is not a bullet/marker
            # 4. At least one ends with lowercase (indicating mid-word break)
            if (last_word and
                first_word and
                last_word[-1].isalpha() and
                first_word[0].isalpha() and
                last_word not in ('*', '-', ':') and
                (last_word[-1].islower() or first_word[0].islower())):

                # Strategy: Check if first_word starts with last_word (case-insensitive)
                # This handles retyped words like "Spi" + "Spironolactone"
                if first_word.lower().startswith(last_word.lower()):
                    # Replace the incomplete word with the complete word
                    # Example: "Furosemide and Spi" + "Spironolactone)" -> "Furosemide and Spironolactone)"
                    prefix = current_line[:last_word_match.start()]
                    merged_lines.append(prefix + next_line)
                else:
                    # Otherwise just join them (true continuation)
                    # Example: "and p" + "ain" -> "and pain"
                    merged_lines.append(current_line + next_line)

                i += 2
                continue

        merged_lines.append(current_line)
        i += 1

    return '\n'.join(merged_lines).strip()
