import os
import re

def _time_to_seconds(time_str):
    """Converts a VTT timestamp string to seconds."""
    time_str = time_str.strip()
    try:
        if '.' in time_str:
            major, ms = time_str.split('.')
        else:
            major, ms = time_str, '0'

        parts = major.split(':')
        s = int(parts[-1])
        m = int(parts[-2]) if len(parts) > 1 else 0
        h = int(parts[-3]) if len(parts) > 2 else 0

        return h * 3600 + m * 60 + s + int(ms) / 1000.0
    except (ValueError, IndexError) as e:
        # Return 0 or raise a more specific error if the format is unexpected
        print(f"Warning: Could not parse timestamp '{time_str}'. Error: {e}")
        return 0.0

def parse_vtt(filepath):
    """
    Parses a VTT file more robustly and returns a list of dictionaries.
    Each dictionary contains 'start', 'end', and 'text'.
    """
    if not filepath or not os.path.exists(filepath):
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lyrics = []
    # Split the file into sections based on double newlines, which typically separate cues
    cues = re.split(r'\n\s*\n', content.strip())

    for cue in cues:
        lines = cue.strip().split('\n')
        if not lines or "WEBVTT" in lines[0]:
            continue

        # Find the line with the timestamp
        time_line_index = -1
        for i, line in enumerate(lines):
            if '-->' in line:
                time_line_index = i
                break

        if time_line_index != -1:
            try:
                timestamp_line = lines[time_line_index]
                start_str, end_str = timestamp_line.split('-->')

                # Clean up end string from potential cue settings
                end_str_cleaned = end_str.strip().split(' ')[0]

                start_time = _time_to_seconds(start_str)
                end_time = _time_to_seconds(end_str_cleaned)

                # The subsequent lines are the text payload
                text_lines = lines[time_line_index + 1:]
                if text_lines:
                    # Join multi-line texts and remove VTT tags
                    full_text = ' '.join(text_lines)
                    cleaned_text = re.sub(r'<[^>]+>', '', full_text).strip()
                    if cleaned_text:
                        lyrics.append({'start': start_time, 'end': end_time, 'text': cleaned_text})
            except (ValueError, IndexError) as e:
                print(f"Skipping malformed VTT cue section: {cue} - Error: {e}")
                continue

    return lyrics