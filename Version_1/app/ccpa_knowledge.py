"""
Built-in CCPA knowledge base — used as the primary source AND as a fallback
when the CCPA statute PDF cannot be parsed at runtime.

Each entry carries:
  • section_id   – e.g. "1798.100"
  • title        – human-readable section name
  • text         – full statutory text (key provisions)
  • page_hint    – approximate page in the official statute PDF
  • keywords     – terms that help the MoE router
  • domain       – expert domain label
"""

from typing import List, Dict, Any

CCPA_SECTIONS: List[Dict[str, Any]] = [
    {
        "section_id": "1798.100",
        "title": "Right to Know About Personal Information Collected",
        "text": (
            "Section 1798.100 - General Rights.\n"
            "(a) A consumer shall have the right to request that a business that "
            "collects a consumer's personal information disclose to that consumer "
            "the categories and specific pieces of personal information the business "
            "has collected.\n"
            "(b) A business that collects a consumer's personal information shall, "
            "at or before the point of collection, inform consumers as to the "
            "categories of personal information to be collected and the purposes "
            "for which the categories of personal information shall be used. A "
            "business shall not collect additional categories of personal information "
            "or use personal information collected for additional purposes without "
            "providing the consumer with notice consistent with this section.\n"
            "(c) A business shall provide the information specified in subdivision "
            "(a) to a consumer only upon receipt of a verifiable consumer request.\n"
            "(d) A business that receives a verifiable consumer request from a "
            "consumer to access personal information shall promptly take steps to "
            "disclose and deliver, free of charge to the consumer, the personal "
            "information required by this section."
        ),
        "page_hint": 2,
        "keywords": [
            "collect", "collection", "disclose", "disclosure", "privacy policy",
            "inform", "notice", "categories", "personal information", "purpose",
            "browsing history", "geolocation", "biometric",
        ],
        "domain": "data_collection",
    },
    {
        "section_id": "1798.105",
        "title": "Consumer's Right to Deletion",
        "text": (
            "Section 1798.105 - Consumers Right to Deletion.\n"
            "(a) A consumer shall have the right to request that a business delete "
            "any personal information about the consumer which the business has "
            "collected from the consumer.\n"
            "(b) A business that collects personal information about consumers "
            "shall disclose, pursuant to Section 1798.130, the consumer's rights "
            "to request the deletion of the consumer's personal information.\n"
            "(c) A business that receives a verifiable consumer request from a "
            "consumer to delete the consumer's personal information pursuant to "
            "subdivision (a) of this section shall delete the consumer's personal "
            "information from its records and direct any service providers to "
            "delete the consumer's personal information from their records.\n"
            "(d) A business or a service provider shall not be required to comply "
            "with a consumer's request to delete the consumer's personal information "
            "if it is necessary for the business or service provider to maintain "
            "the consumer's personal information in order to: (1) Complete the "
            "transaction for which the personal information was collected; "
            "(2) Detect security incidents; (3) Exercise free speech; "
            "(4) Comply with the California Electronic Communications Privacy Act; "
            "(5) Engage in research in the public interest; (6) Enable internal uses "
            "reasonably aligned with consumer expectations; (7) Comply with a legal "
            "obligation; (8) Otherwise use the consumer's personal information, "
            "internally, in a lawful manner compatible with the context in which "
            "the consumer provided the information."
        ),
        "page_hint": 3,
        "keywords": [
            "delete", "deletion", "erase", "remove", "request to delete",
            "right to delete", "ignore", "refuse deletion", "keep records",
        ],
        "domain": "deletion_rights",
    },
    {
        "section_id": "1798.110",
        "title": "Right to Know What Personal Information is Being Collected",
        "text": (
            "Section 1798.110 - Right to Know What is Collected.\n"
            "(a) A consumer shall have the right to request that a business that "
            "collects personal information about the consumer disclose to the "
            "consumer the following: (1) The categories of personal information "
            "it has collected about that consumer. (2) The categories of sources "
            "from which the personal information is collected. (3) The business or "
            "commercial purpose for collecting or selling personal information. "
            "(4) The categories of third parties with whom the business shares "
            "personal information. (5) The specific pieces of personal information "
            "it has collected about that consumer.\n"
            "(b) A business that collects personal information about a consumer "
            "shall disclose to the consumer, pursuant to subparagraph (B) of "
            "paragraph (3) of subdivision (a) of Section 1798.130, the information "
            "specified in subdivision (a) upon receipt of a verifiable consumer request."
        ),
        "page_hint": 4,
        "keywords": [
            "categories", "sources", "purpose", "third parties",
            "what information", "collected about",
        ],
        "domain": "data_collection",
    },
    {
        "section_id": "1798.115",
        "title": "Right to Know About Selling or Disclosure for Business Purpose",
        "text": (
            "Section 1798.115 - Disclosure of Sale or Sharing.\n"
            "(a) A consumer shall have the right to request that a business that "
            "sells the consumer's personal information, or that discloses it for "
            "a business purpose, disclose to that consumer: (1) The categories of "
            "personal information that the business collected about the consumer. "
            "(2) The categories of personal information that the business sold about "
            "the consumer and the categories of third parties to whom the personal "
            "information was sold, by category or categories of personal information "
            "for each third party to whom the personal information was sold. "
            "(3) The categories of personal information that the business disclosed "
            "about the consumer for a business purpose.\n"
            "(b) A business that sells personal information about a consumer, or "
            "that discloses a consumer's personal information for a business purpose, "
            "shall disclose, pursuant to Section 1798.130, the information specified "
            "in subdivision (a)."
        ),
        "page_hint": 5,
        "keywords": [
            "sell", "sale", "disclose", "business purpose", "third party",
            "sharing", "sold", "categories sold",
        ],
        "domain": "opt_out_sale",
    },
    {
        "section_id": "1798.120",
        "title": "Consumer's Right to Opt-Out of Sale of Personal Information",
        "text": (
            "Section 1798.120 - Right to Opt-Out.\n"
            "(a) A consumer shall have the right, at any time, to direct a business "
            "that sells personal information about the consumer to third parties "
            "not to sell the consumer's personal information. This right may be "
            "referred to as the right to opt-out.\n"
            "(b) A business that sells consumers' personal information to third "
            "parties shall provide notice to consumers, pursuant to subdivision (a) "
            "of Section 1798.135, that this information may be sold and that "
            "consumers have the right to opt-out of the sale of their personal "
            "information.\n"
            "(c) Notwithstanding subdivision (a), a business shall not sell the "
            "personal information of consumers if the business has actual knowledge "
            "that the consumer is less than 16 years of age, unless the consumer, "
            "in the case of consumers between 13 and 16 years of age, has "
            "affirmatively authorized the sale of the consumer's personal "
            "information. A business that willfully disregards the consumer's age "
            "shall be deemed to have had actual knowledge of the consumer's age. "
            "This right may be referred to as the right to opt-in.\n"
            "(d) A business that has received direction from a consumer not to sell "
            "the consumer's personal information or, in the case of a minor "
            "consumer's personal information has not received consent to sell the "
            "minor consumer's personal information, shall be prohibited from selling "
            "the consumer's personal information after its receipt of the consumer's "
            "direction, unless the consumer subsequently provides express "
            "authorization for the sale of the consumer's personal information.\n"
            "For consumers less than 13 years of age, a parent or guardian must "
            "provide verifiable consent for the sale of personal information."
        ),
        "page_hint": 6,
        "keywords": [
            "sell", "sale", "opt-out", "opt out", "third party", "broker",
            "minor", "child", "children", "under 16", "under 13", "parental consent",
            "14-year", "teenager", "age",
        ],
        "domain": "opt_out_sale",
    },
    {
        "section_id": "1798.125",
        "title": "Consumer's Right of Non-Discrimination",
        "text": (
            "Section 1798.125 - Non-Discrimination.\n"
            "(a) A business shall not discriminate against a consumer because the "
            "consumer exercised any of the consumer's rights under this title, "
            "including, but not limited to, by: (1) Denying goods or services to "
            "the consumer. (2) Charging different prices or rates for goods or "
            "services, including through the use of discounts or other benefits or "
            "imposing penalties. (3) Providing a different level or quality of "
            "goods or services to the consumer. (4) Suggesting that the consumer "
            "will receive a different price or rate for goods or services or a "
            "different level or quality of goods or services.\n"
            "(b) Nothing in this section prohibits a business from charging a "
            "consumer a different price or rate, or from providing a different "
            "level or quality of goods or services to the consumer, if that "
            "difference is reasonably related to the value provided to the consumer "
            "by the consumer's data.\n"
            "(c) A business may offer financial incentives, including payments to "
            "consumers as compensation, for the collection of personal information, "
            "the sale of personal information, or the deletion of personal "
            "information. A business may also offer a different price, rate, level, "
            "or quality of goods or services to the consumer if that price or "
            "difference is directly related to the value provided to the consumer "
            "by the consumer's data."
        ),
        "page_hint": 7,
        "keywords": [
            "discriminat", "price", "pricing", "penalty", "deny", "service",
            "quality", "different rate", "punish", "retaliat", "higher price",
            "opted out", "exercise rights",
        ],
        "domain": "non_discrimination",
    },
    {
        "section_id": "1798.130",
        "title": "Notice and Response Requirements",
        "text": (
            "Section 1798.130 - Notice, Disclosure, and Response Requirements.\n"
            "(a) In order to comply with Sections 1798.100, 1798.105, 1798.110, "
            "and 1798.115, a business shall, in a form that is reasonably accessible "
            "to consumers: (1) Make available to consumers two or more designated "
            "methods for submitting requests for information required to be "
            "disclosed, including, at a minimum, a toll-free telephone number. "
            "(2) Disclose and deliver the required information to a consumer free "
            "of charge within 45 days of receiving a verifiable consumer request. "
            "The time period may be extended once by an additional 45 days when "
            "reasonably necessary. (3) For purposes of subdivision (b) of Section "
            "1798.110: (A) identify the consumer and associate the information "
            "provided; (B) provide the required information; (C) not be required "
            "to re-identify or otherwise link information not maintained in a "
            "manner associated with an identified or identifiable consumer. "
            "(4) A business that receives a verifiable consumer request to delete "
            "information shall also inform the consumer that the request to delete "
            "will result in the deletion of their personal information."
        ),
        "page_hint": 8,
        "keywords": [
            "notice", "response", "45 days", "toll-free", "telephone",
            "methods", "submitting requests", "disclose", "deliver",
        ],
        "domain": "data_collection",
    },
    {
        "section_id": "1798.135",
        "title": "Methods for Submitting Opt-Out Requests",
        "text": (
            "Section 1798.135 - Opt-Out Methods.\n"
            "(a) A business that is required to comply with Section 1798.120 shall, "
            "in a form that is reasonably accessible to consumers: (1) Provide a "
            "clear and conspicuous link on the business's Internet homepage, titled "
            "'Do Not Sell My Personal Information,' to an Internet Web page that "
            "enables a consumer, or a person authorized by the consumer, to opt "
            "out of the sale of the consumer's personal information. A business "
            "shall not require a consumer to create an account in order to direct "
            "the business not to sell the consumer's personal information. "
            "(2) Include a description of a consumer's rights pursuant to Section "
            "1798.120, along with a separate link to the 'Do Not Sell My Personal "
            "Information' Internet Web page in: (A) its online privacy policy or "
            "policies if the business has an online privacy policy or policies; and "
            "(B) in any California-specific description of consumers' privacy rights."
        ),
        "page_hint": 9,
        "keywords": [
            "do not sell", "homepage", "link", "opt-out page", "conspicuous",
            "internet", "web page", "create account",
        ],
        "domain": "opt_out_sale",
    },
    {
        "section_id": "1798.140",
        "title": "Definitions",
        "text": (
            "Section 1798.140 - Definitions.\n"
            "Key definitions include:\n"
            "(a) 'Aggregate consumer information' means information that relates "
            "to a group or category of consumers, from which individual consumer "
            "identities have been removed.\n"
            "(c) 'Business' means a sole proprietorship, partnership, LLC, "
            "corporation, association, or other legal entity that is organized or "
            "operated for the profit or financial benefit of its shareholders or "
            "other owners.\n"
            "(d) 'Business purpose' means the use of personal information for the "
            "business's or a service provider's operational purposes.\n"
            "(g) 'Consumer' means a natural person who is a California resident.\n"
            "(o) 'Personal information' means information that identifies, relates "
            "to, describes, is reasonably capable of being associated with, or "
            "could reasonably be linked, directly or indirectly, with a particular "
            "consumer or household. Personal information includes, but is not "
            "limited to: (1) Identifiers such as a real name, alias, postal "
            "address, unique personal identifier, online identifier; (2) Commercial "
            "information including records of personal property, products or "
            "services purchased, obtained, or considered; (3) Biometric information; "
            "(4) Internet or other electronic network activity information, "
            "including browsing history, search history, and information regarding "
            "a consumer's interaction with an Internet Web site, application, or "
            "advertisement; (5) Geolocation data; (6) Audio, electronic, visual, "
            "thermal, olfactory, or similar information; (7) Professional or "
            "employment-related information; (8) Education information.\n"
            "(t) 'Sell,' 'selling,' 'sale,' or 'sold' means selling, renting, "
            "releasing, disclosing, disseminating, making available, transferring, "
            "or otherwise communicating orally, in writing, or by electronic or "
            "other means, a consumer's personal information by the business to "
            "another business or a third party for monetary or other valuable "
            "consideration."
        ),
        "page_hint": 10,
        "keywords": [
            "definition", "personal information", "business", "consumer",
            "sell", "service provider", "aggregate", "biometric",
            "browsing history", "geolocation",
        ],
        "domain": "general_compliance",
    },
    {
        "section_id": "1798.145",
        "title": "Exemptions",
        "text": (
            "Section 1798.145 - Exemptions.\n"
            "(a) The obligations imposed on businesses by this title shall not "
            "restrict a business's ability to: (1) Comply with federal, state, or "
            "local laws. (2) Comply with a civil, criminal, or regulatory inquiry. "
            "(3) Cooperate with law enforcement agencies concerning conduct or "
            "activity that the business reasonably and in good faith believes may "
            "violate federal, state, or local law. (4) Exercise or defend legal "
            "claims. (5) Collect, use, retain, sell, or disclose consumer "
            "information that is de-identified or in the aggregate consumer "
            "information. (6) Collect or sell a consumer's personal information "
            "if every aspect of that commercial conduct takes place wholly outside "
            "of California."
        ),
        "page_hint": 12,
        "keywords": [
            "exempt", "exception", "comply", "law enforcement",
            "de-identified", "aggregate", "outside California",
        ],
        "domain": "general_compliance",
    },
    {
        "section_id": "1798.150",
        "title": "Personal Information Security Breaches",
        "text": (
            "Section 1798.150 - Personal Right of Action for Data Breaches.\n"
            "(a) Any consumer whose nonencrypted or nonredacted personal "
            "information is subject to an unauthorized access and exfiltration, "
            "theft, or disclosure as a result of the business's violation of the "
            "duty to implement and maintain reasonable security procedures and "
            "practices appropriate to the nature of the information may institute "
            "a civil action for any of the following: (1) To recover damages in "
            "an amount not less than one hundred dollars ($100) and not greater "
            "than seven hundred and fifty ($750) per consumer per incident or "
            "actual damages, whichever is greater. (2) Injunctive or declaratory "
            "relief. (3) Any other relief the court deems proper."
        ),
        "page_hint": 13,
        "keywords": [
            "breach", "security", "unauthorized access", "theft",
            "encryption", "civil action", "damages",
        ],
        "domain": "general_compliance",
    },
    {
        "section_id": "1798.155",
        "title": "Administrative Enforcement",
        "text": (
            "Section 1798.155 - Administrative Enforcement.\n"
            "(a) Any business or third party may be liable for a civil penalty "
            "of not more than two thousand five hundred dollars ($2,500) for each "
            "violation, or seven thousand five hundred dollars ($7,500) for each "
            "intentional violation, which shall be assessed and recovered in a "
            "civil action brought in the name of the people of the State of "
            "California by the Attorney General.\n"
            "(b) Any civil penalty assessed for a violation of this title, and "
            "the proceeds of any settlement of an action brought pursuant to "
            "subdivision (a), shall be deposited in the Consumer Privacy Fund."
        ),
        "page_hint": 14,
        "keywords": [
            "penalty", "enforcement", "attorney general", "fine",
            "violation", "civil action",
        ],
        "domain": "general_compliance",
    },
]

# ── Quick-lookup helpers ───────────────────────────────────────────────
SECTION_MAP: Dict[str, Dict[str, Any]] = {s["section_id"]: s for s in CCPA_SECTIONS}
DOMAIN_SECTIONS: Dict[str, List[str]] = {}
for _s in CCPA_SECTIONS:
    DOMAIN_SECTIONS.setdefault(_s["domain"], []).append(_s["section_id"])


def get_sections_by_domain(domain: str) -> List[Dict[str, Any]]:
    """Return all sections belonging to a given expert domain."""
    return [SECTION_MAP[sid] for sid in DOMAIN_SECTIONS.get(domain, [])]


def get_all_section_texts() -> List[Dict[str, Any]]:
    """Return lightweight dicts suited for the chunker."""
    return [
        {
            "section_id": s["section_id"],
            "title": s["title"],
            "text": s["text"],
            "page": s["page_hint"],
            "domain": s["domain"],
        }
        for s in CCPA_SECTIONS
    ]
