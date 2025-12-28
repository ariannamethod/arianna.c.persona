"""Categorize books into core (concepts) and stories"""

from pathlib import Path
import shutil

personality_dir = Path('personality')
core_dir = personality_dir / 'core'
stories_dir = personality_dir / 'stories'

# Create directories
core_dir.mkdir(exist_ok=True)
stories_dir.mkdir(exist_ok=True)

# Categorize books
books = list(personality_dir.glob('*.md'))
print(f"Found {len(books)} books to categorize")

core_count = 0
stories_count = 0

for book in books:
    with open(book, 'r') as f:
        first_line = f.readline().strip()

    if first_line.startswith('ARIANNA:'):
        # Conceptual book → core
        dest = core_dir / book.name
        shutil.move(str(book), str(dest))
        core_count += 1
        print(f"CORE: {book.name} → {first_line}")
    elif first_line.startswith('THE STORY'):
        # Story book → stories
        dest = stories_dir / book.name
        shutil.move(str(book), str(dest))
        stories_count += 1
    else:
        print(f"UNKNOWN: {book.name} → {first_line}")

print(f"\n✓ Categorized {core_count} core books")
print(f"✓ Categorized {stories_count} story books")
