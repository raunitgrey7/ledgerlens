"""Catalogues of fictional names, addresses, products and transaction descriptions.

Everything here is invented so no real company, person or account appears in the
dataset. The lists are deliberately varied in length and character mix (digits,
slashes, ampersands) because those are the strings that break naive OCR tokenisers.
"""

from __future__ import annotations

from dataclasses import dataclass

COMPANY_HEADS: tuple[str, ...] = (
    "Meridian",
    "Northgate",
    "Blue Lotus",
    "Kaveri",
    "Harbour",
    "Silverline",
    "Ashoka",
    "Granite",
    "Westfield",
    "Aurora",
    "Cedar",
    "Trident",
    "Sunrise",
    "Pioneer",
    "Oakridge",
    "Vistara",
    "Larkspur",
    "Ironwood",
    "Saffron",
    "Pelican",
    "Kestrel",
    "Marigold",
    "Summit",
    "Tulsi",
)
COMPANY_TRADES: tuple[str, ...] = (
    "Textiles",
    "Logistics",
    "Engineering",
    "Software",
    "Consulting",
    "Traders",
    "Electricals",
    "Agro Foods",
    "Packaging",
    "Printers",
    "Hardware",
    "Pharma",
    "Interiors",
    "Analytics",
    "Builders",
    "Exports",
    "Stationers",
    "Machines",
    "Chemicals",
    "Media",
)
COMPANY_SUFFIX: dict[str, tuple[str, ...]] = {
    "IN": ("Pvt Ltd", "Private Limited", "LLP", "& Sons", "Enterprises", "Industries"),
    "GB": ("Ltd", "Limited", "LLP", "& Co", "Group Ltd"),
    "DE": ("GmbH", "AG", "KG", "GmbH & Co. KG"),
    "US": ("Inc", "LLC", "Corp", "Co."),
}
RETAIL_HEADS: tuple[str, ...] = (
    "Sharma General Store",
    "Daily Needs Mart",
    "Greenmart",
    "Corner Bakery",
    "Cafe Nimbus",
    "Quick Stop",
    "Fresh Basket",
    "Nordkauf Markt",
    "Riverside Pharmacy",
    "Lakshmi Sweets",
    "Bean & Leaf",
    "Urban Grocer",
    "Spice Route Kitchen",
    "Hilltop Books",
    "Metro Fuel Point",
)
BANK_NAMES: dict[str, tuple[str, ...]] = {
    "IN": ("Saraswati Bank", "Indus Valley Bank", "Peninsula Bank Ltd"),
    "GB": ("Albion Bank plc", "Thames & Mersey Bank", "Northern Rock Lane Bank"),
    "DE": ("Rheinbank AG", "Hansa Sparkasse", "Schwarzwald Volksbank"),
    "US": ("Prairie National Bank", "Harbor Trust Bank", "Redwood Federal Credit Union"),
}
PERSON_NAMES: tuple[str, ...] = (
    "Priya Raman",
    "Arjun Mehta",
    "Neha Kulkarni",
    "Rohit Verma",
    "Ananya Iyer",
    "Vikram Nair",
    "James Whitfield",
    "Emma Clarke",
    "Oliver Hughes",
    "Sophie Turner",
    "Lukas Brandt",
    "Anna Schreiber",
    "Jonas Keller",
    "Mia Hoffmann",
    "Daniel Price",
    "Olivia Bennett",
    "Carlos Ortega",
    "Hannah Weiss",
    "Ravi Shankar",
    "Meera Pillai",
)


@dataclass(frozen=True)
class Address:
    """Postal address lines as printed on a document."""

    street: str
    city: str
    postcode: str
    country: str

    def lines(self) -> list[str]:
        """Two printable lines in the local convention."""
        if self.country == "US":
            return [self.street, f"{self.city}, {self.postcode}"]
        if self.country == "DE":
            return [self.street, f"{self.postcode} {self.city}"]
        return [self.street, f"{self.city} {self.postcode}"]


STREETS: dict[str, tuple[str, ...]] = {
    "IN": (
        "14 MG Road",
        "221 Nehru Place",
        "7 Linking Road",
        "88 Anna Salai",
        "45 Brigade Road",
        "3rd Floor, Sector 18",
        "Plot 12, MIDC Andheri",
        "19 Park Street",
    ),
    "GB": ("12 High Street", "4 Kings Road", "77 Station Approach", "9 Mill Lane", "31 Castle Street"),
    "DE": ("Hauptstrasse 12", "Bahnhofstrasse 7", "Lindenallee 23", "Marktplatz 3", "Ringstrasse 45"),
    "US": ("1200 Market St", "48 Elm Avenue", "901 Lakeshore Dr", "77 Pine Street", "350 5th Ave"),
}
CITIES: dict[str, tuple[tuple[str, str], ...]] = {
    "IN": (
        ("New Delhi", "110001"),
        ("Mumbai", "400053"),
        ("Bengaluru", "560001"),
        ("Chennai", "600002"),
        ("Pune", "411001"),
        ("Hyderabad", "500081"),
        ("Kolkata", "700016"),
        ("Jaipur", "302001"),
    ),
    "GB": (("London", "SW1A 1AA"), ("Manchester", "M1 2AB"), ("Leeds", "LS1 4AP"), ("Bristol", "BS1 5TR")),
    "DE": (("Berlin", "10115"), ("Hamburg", "20095"), ("Munich", "80331"), ("Frankfurt", "60311")),
    "US": (("Austin, TX", "78701"), ("Denver, CO", "80202"), ("Seattle, WA", "98101"), ("Boston, MA", "02108")),
}
CURRENCY: dict[str, str] = {"IN": "INR", "GB": "GBP", "DE": "EUR", "US": "USD"}
CURRENCY_SYMBOL: dict[str, str] = {"INR": "₹", "GBP": "£", "EUR": "€", "USD": "$"}
# Rough purchasing-power scale so that a "medium" item costs a plausible local amount.
PRICE_SCALE: dict[str, float] = {"INR": 80.0, "GBP": 0.8, "EUR": 0.95, "USD": 1.0}

# (description, base price in USD-equivalent units, is_service)
PRODUCTS: tuple[tuple[str, float, bool], ...] = (
    ("Steel bracket 40mm", 4.5, False),
    ("Copper wire 2.5 sqmm (100m)", 38.0, False),
    ("A4 copier paper 80gsm", 6.0, False),
    ("Laser toner cartridge", 72.0, False),
    ("Office chair ergonomic", 185.0, False),
    ("LED panel light 36W", 21.0, False),
    ("Cloud hosting (monthly)", 120.0, True),
    ("Consulting hours", 95.0, True),
    ("Software licence seat", 49.0, True),
    ("Freight charges", 60.0, True),
    ("Installation service", 150.0, True),
    ("Annual maintenance contract", 480.0, True),
    ("Cotton fabric 60in (metre)", 3.2, False),
    ("Packaging carton 18x12x10", 1.4, False),
    ("Industrial adhesive 5kg", 27.0, False),
    ("Safety gloves (pair)", 2.6, False),
    ("Network switch 24 port", 210.0, False),
    ("USB-C cable 1m", 8.0, False),
    ("Training workshop (day)", 650.0, True),
    ("Design revision", 140.0, True),
    ("Hex bolts M8 (box of 100)", 9.5, False),
    ("Printer maintenance kit", 88.0, False),
    ("Thermal paper roll 80mm", 1.1, False),
    ("Water pump 1HP", 165.0, False),
    ("Marketing retainer", 900.0, True),
    ("Site survey", 300.0, True),
)
RETAIL_ITEMS: tuple[tuple[str, float], ...] = (
    ("Milk 1L", 1.1),
    ("Brown bread", 1.6),
    ("Basmati rice 5kg", 9.0),
    ("Toor dal 1kg", 2.4),
    ("Cappuccino", 3.2),
    ("Masala chai", 1.0),
    ("Croissant", 2.1),
    ("Veg sandwich", 3.8),
    ("Mineral water 500ml", 0.5),
    ("Paracetamol 10 tabs", 0.8),
    ("Hand wash 250ml", 2.2),
    ("Notebook A5", 1.9),
    ("Ballpoint pen", 0.4),
    ("Bananas 1kg", 1.2),
    ("Tomatoes 500g", 0.9),
    ("Sunflower oil 1L", 2.8),
    ("Biscuits 200g", 0.7),
    ("Shampoo 180ml", 3.4),
    ("Eggs 12", 2.6),
    ("Orange juice 1L", 2.3),
    ("Chocolate bar", 1.3),
    ("Dish soap 500ml", 1.7),
)
TRANSACTION_TEMPLATES: dict[str, tuple[tuple[str, str], ...]] = {
    # (description pattern, kind) where kind is "debit", "credit" or "either"
    "IN": (
        ("UPI/{ref}/Foodly", "debit"),
        ("POS {ref} GREENMART", "debit"),
        ("ATM WDL/DEL{ref4}", "debit"),
        ("NEFT/AXN{ref}/MERIDIAN TEXTILES", "credit"),
        ("SALARY CREDIT {company}", "credit"),
        ("IMPS/{ref}/RENT", "debit"),
        ("INT CREDIT", "credit"),
        ("EMI {company} LOAN", "debit"),
        ("UPI/{ref}/METRO FUEL", "debit"),
        ("BILL PAY ELECTRICITY {ref4}", "debit"),
    ),
    "GB": (
        ("CARD PAYMENT TO GREENMART", "debit"),
        ("DIRECT DEBIT ALBION ENERGY", "debit"),
        ("FASTER PAYMENT FROM {person}", "credit"),
        ("SALARY {company}", "credit"),
        ("CASH WITHDRAWAL HIGH ST", "debit"),
        ("STANDING ORDER RENT", "debit"),
        ("CONTACTLESS CORNER BAKERY", "debit"),
        ("INTEREST PAID", "credit"),
    ),
    "DE": (
        ("LASTSCHRIFT STADTWERKE", "debit"),
        ("KARTENZAHLUNG NORDKAUF", "debit"),
        ("GEHALT {company}", "credit"),
        ("UEBERWEISUNG {person}", "either"),
        ("BARGELDAUSZAHLUNG", "debit"),
        ("MIETE WOHNUNG", "debit"),
        ("ZINSGUTSCHRIFT", "credit"),
    ),
    "US": (
        ("POS PURCHASE FRESH BASKET", "debit"),
        ("ACH DEBIT HARBOR CABLE", "debit"),
        ("DIRECT DEP PAYROLL {company}", "credit"),
        ("ZELLE FROM {person}", "credit"),
        ("ATM WITHDRAWAL #{ref4}", "debit"),
        ("CHECK {ref4}", "debit"),
        ("INTEREST CREDIT", "credit"),
    ),
}
