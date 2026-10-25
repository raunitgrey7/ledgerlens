"""ledgerlens: document intelligence for invoices, receipts and bank statements.

The package renders its own labelled documents, reads them back with a deterministic
OCR and extraction pipeline, validates the result with accounting rules, and routes
low-confidence pages to a review queue. Everything runs offline and seeded.
"""

__version__ = "0.1.0"


def main() -> None:
    """Console entry point; the real work lives in ``ledgerlens.cli``."""
    from ledgerlens.cli import run

    run()
