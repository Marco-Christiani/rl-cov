import inspect
import re
import webbrowser
from pathlib import Path

import pandas as pd
import yaml

import rlcov

research_dir = Path(__file__).parent
print(research_dir)
yaml_style_key = 'row_style'


def extract_frontmatter(file_path: Path) -> dict:
    try:
        content = file_path.read_text(encoding='utf-8')
    except IOError:
        return {}

    pattern = r"^---(.*?)^---"
    match = re.search(pattern, content, re.DOTALL | re.MULTILINE)

    if not match:
        raise ValueError("Frontmatter not found in the file.")

    frontmatter = match.group(1).strip()

    try:
        return yaml.safe_load(frontmatter)
    except yaml.YAMLError:
        return {}


def format_list_for_html(value):
    if isinstance(value, list):
        # return str('\n- '.join(map(str, value)))
        html_list = ''.join(f"<li>{item}</li>" for item in value)
        return f"<ul>{html_list}</ul>"
    return value


def custom_row_style(row):
    style = row.get(yaml_style_key, '')
    if style:
        return [style] * len(row)  # Apply custom style to all cells in the row
    return [''] * len(row)


def compile_frontmatter_to_table(directory: Path, use_styles: bool = False) -> str:
    frontmatter_list = []
    for f in directory.glob('*.md'):
        frontmatter_list.append({'filename': f'<a href={f.relative_to(research_dir)}>{f.name}</a>'}
                                | extract_frontmatter(f))

    df = pd.DataFrame(frontmatter_list)
    df = df.applymap(format_list_for_html)
    styler = df.style
    if use_styles:
        styler.set_table_styles([{
            'selector': 'th, td',
            'props': [
                ('word-wrap', 'break-word'),
                ('max-width', '350px')
            ]
        }])
    if yaml_style_key in df.columns:
        styler = styler.apply(custom_row_style, axis=1).hide(yaml_style_key, axis=1)
    styler = styler.hide('doi', axis=1)
    return styler.to_html(index=False, exclude_styles=not use_styles)


if __name__ == "__main__":
    html = compile_frontmatter_to_table(research_dir/'papers')
    print(html)
    md_content = '# Literature Review\n\n' + html
    with open(research_dir/'LiteratureReview.md', 'wb') as f:
        f.write(md_content.encode('utf-8'))
