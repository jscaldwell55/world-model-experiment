#!/usr/bin/env python3
"""
Convert RESULTS_SUMMARY.md to a professionally-styled HTML file.

This script converts the experiment results summary from Markdown to HTML
with professional styling optimized for PDF export via browser print.

Usage:
    python create_html.py

Output:
    RESULTS_SUMMARY.html - Styled HTML ready for browser-based PDF export

To create PDF:
    1. Open RESULTS_SUMMARY.html in a web browser
    2. Print to PDF (Cmd+P on Mac, Ctrl+P on Windows)
    3. Save as RESULTS_SUMMARY.pdf
"""

import markdown
from pathlib import Path

# Read the markdown file
md_file = Path("RESULTS_SUMMARY.md")
if not md_file.exists():
    print(f"❌ Error: {md_file} not found")
    exit(1)

md_content = md_file.read_text()

# Convert markdown to HTML
html_body = markdown.markdown(
    md_content,
    extensions=['tables', 'fenced_code']
)

# Create CSS styling
css_style = """
    @page {
        size: Letter;
        margin: 0.75in;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 900px;
        margin: 0 auto;
        padding: 20px;
        font-size: 11pt;
    }

    h1 {
        color: #1a1a1a;
        border-bottom: 3px solid #0066cc;
        padding-bottom: 10px;
        margin-top: 0;
        font-size: 28pt;
    }

    h2 {
        color: #2c3e50;
        border-bottom: 2px solid #95a5a6;
        padding-bottom: 8px;
        margin-top: 30px;
        font-size: 18pt;
        page-break-after: avoid;
    }

    h3 {
        color: #34495e;
        margin-top: 20px;
        font-size: 14pt;
        page-break-after: avoid;
    }

    table {
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
        page-break-inside: avoid;
        font-size: 10pt;
    }

    th, td {
        border: 1px solid #ddd;
        padding: 10px;
        text-align: left;
    }

    th {
        background-color: #0066cc;
        color: white;
        font-weight: 600;
    }

    tr:nth-child(even) {
        background-color: #f8f9fa;
    }

    code {
        background-color: #f4f4f4;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: "Monaco", "Courier New", monospace;
        font-size: 0.9em;
        color: #d63384;
    }

    pre {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #0066cc;
        overflow-x: auto;
        page-break-inside: avoid;
    }

    pre code {
        background-color: transparent;
        padding: 0;
        color: inherit;
    }

    strong {
        color: #1a1a1a;
        font-weight: 600;
    }

    ul, ol {
        margin: 15px 0;
        padding-left: 30px;
    }

    li {
        margin: 8px 0;
    }

    hr {
        border: none;
        border-top: 2px solid #e9ecef;
        margin: 30px 0;
    }

    .footer {
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #e9ecef;
        font-size: 9pt;
        color: #6c757d;
        text-align: center;
    }

    @media print {
        body {
            padding: 0;
        }

        h2 {
            page-break-before: auto;
        }
    }
"""

# Create the final HTML
full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>World Model Experiment Results Summary</title>
    <style>
{css_style}
    </style>
</head>
<body>
    {html_body}

    <div class="footer">
        Generated from RESULTS_SUMMARY.md | World Model Experiment | 2025
    </div>
</body>
</html>
"""

# Write to file
output_file = "RESULTS_SUMMARY.html"
Path(output_file).write_text(full_html)

file_size_kb = Path(output_file).stat().st_size / 1024
print(f"✅ Successfully created {output_file}")
print(f"📄 File size: {file_size_kb:.1f} KB")
print()
print("📝 To create PDF:")
print("   1. Open RESULTS_SUMMARY.html in your browser")
print("   2. Press Cmd+P (Mac) or Ctrl+P (Windows)")
print("   3. Select 'Save as PDF' as the destination")
print("   4. Save as RESULTS_SUMMARY.pdf")
print()
print("The PDF will be professionally formatted and ready to share!")
