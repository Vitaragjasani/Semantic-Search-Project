"""Generates the demo documents under sample_docs/ for Phase 10.

Run once from the project root (after `pip install -r requirements.txt`):
    python scripts/generate_sample_docs.py

Kept as a script rather than committing only the binary outputs so the
demo content's source text is reviewable and regeneratable -- the PDF and
DOCX files it produces are still committed too, since they're what you'd
actually drag into the Streamlit uploader.
"""
from pathlib import Path

import docx
import fitz

SAMPLE_DOCS_DIR = Path(__file__).resolve().parent.parent / "sample_docs"


def write_refund_policy_markdown():
    content = """# Refund and Exchange Policy

## Refunds

Refunds are processed within 10 business days of us receiving the returned item. \
The refund is issued to the original payment method. Shipping costs are non-refundable \
unless the return is due to our error.

## Exchanges

Exchanges for a different size or color are processed within 5 business days. \
Exchanged items ship free of charge within the continental US. International \
exchanges may incur additional shipping fees.

## Eligibility

Items must be returned within 30 days of delivery, unused, and in their original \
packaging. Final sale items are not eligible for refund or exchange.
"""
    (SAMPLE_DOCS_DIR / "refund_policy.md").write_text(content, encoding="utf-8")


def write_product_catalog_txt():
    content = """Product Catalog Excerpt - Replacement Parts

Part number ACME-4471-X is the replacement cooling fan assembly for the \
Model 200 series units. It is compatible with units manufactured after 2022 \
and includes a 2-year limited warranty.

Part number ACME-3310-B is the replacement power supply unit for the Model 100 \
series. It is not compatible with Model 200 series units.

For installation instructions, contact technical support and reference the \
part number above.
"""
    (SAMPLE_DOCS_DIR / "product_catalog.txt").write_text(content, encoding="utf-8")


def write_employee_handbook_pdf():
    pdf = fitz.open()

    page1 = pdf.new_page()
    page1.insert_text(
        (72, 72),
        "Employee Handbook\n\n"
        "Section 1: Welcome\n\n"
        "Welcome to the company. This handbook covers our core policies on\n"
        "time off, conduct, and benefits. Please read it in full and reach\n"
        "out to HR with any questions.",
        fontsize=11,
    )

    page2 = pdf.new_page()
    page2.insert_text(
        (72, 72),
        "Section 2: Paid Time Off\n\n"
        "New employees accrue 15 paid vacation days per year, credited\n"
        "monthly starting from their hire date. Unused vacation days roll\n"
        "over up to a maximum of 5 days into the following year. Sick leave\n"
        "is accrued separately at 1 day per month.",
        fontsize=11,
    )

    pdf.save(SAMPLE_DOCS_DIR / "employee_handbook.pdf")
    pdf.close()


def write_onboarding_notes_docx():
    document = docx.Document()
    document.add_paragraph("Onboarding Notes for New Hires")
    document.add_paragraph(
        "All new hires complete a 3-day onboarding program covering company "
        "tools, security training, and team introductions."
    )
    document.add_paragraph(
        "IT equipment (laptop, monitor, headset) is provisioned before the "
        "start date and should arrive within 2 business days of offer acceptance."
    )
    document.add_paragraph(
        "Questions about onboarding logistics should go to the People Operations "
        "team, not the hiring manager."
    )
    document.save(SAMPLE_DOCS_DIR / "onboarding_notes.docx")


def main():
    SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    write_refund_policy_markdown()
    write_product_catalog_txt()
    write_employee_handbook_pdf()
    write_onboarding_notes_docx()
    print(f"Sample documents written to {SAMPLE_DOCS_DIR}")


if __name__ == "__main__":
    main()
