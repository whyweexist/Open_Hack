"""
Pre-parsed CCPA sections with:
  - section number, title, full summary
  - violation criteria (what constitutes a breach)
  - keywords for hybrid retrieval
  - page_range from the PDF (populated by pdf_processor at startup)

This serves as the PRIMARY knowledge source so the system works
even if PDF parsing fails.
"""

CCPA_SECTIONS = {
    "Section 1798.100": {
        "title": "General Duties of Businesses that Collect Personal Information",
        "text": (
            "Section 1798.100 requires that a business that collects a consumer's "
            "personal information shall, at or before the point of collection, inform "
            "consumers as to the categories of personal information to be collected and "
            "the purposes for which the categories of personal information are collected "
            "or used and whether that information is sold or shared. A business shall not "
            "collect additional categories of personal information or use personal "
            "information collected for additional purposes that are incompatible with the "
            "disclosed purpose without providing the consumer with notice. A business "
            "that collects a consumer's personal information shall, not later than at the "
            "point of collection, provide notice to the consumer identifying the "
            "categories of personal information collected and the purposes. A business's "
            "collection, use, retention, and sharing of a consumer's personal information "
            "shall be reasonably necessary and proportionate to achieve the purposes for "
            "which the personal information was collected or processed."
        ),
        "violation_criteria": [
            "Collecting personal information without informing consumers",
            "Not disclosing categories of personal information collected",
            "Not disclosing the purposes of collection",
            "Using personal information for undisclosed or incompatible purposes",
            "Collecting more data than reasonably necessary",
            "Failing to provide notice at or before the point of collection",
            "Retaining personal information longer than necessary",
            "Not disclosing whether information is sold or shared",
        ],
        "keywords": [
            "collect", "notice", "purpose", "disclose", "inform", "categories",
            "privacy policy", "notification", "consent", "proportionate",
            "necessary", "retention", "use", "processing", "data collection",
            "without telling", "without informing", "without notifying",
            "secretly", "covert", "hidden",
        ],
        "page_range": None,
    },
    "Section 1798.105": {
        "title": "Consumers' Right to Delete Personal Information",
        "text": (
            "Section 1798.105 provides that a consumer shall have the right to request "
            "that a business delete any personal information about the consumer which the "
            "business has collected from the consumer. A business that collects personal "
            "information about consumers shall disclose the consumer's right to request "
            "deletion. A business that receives a verifiable consumer request to delete "
            "the consumer's personal information shall delete the consumer's personal "
            "information from its records, notify any service providers or contractors to "
            "delete the consumer's personal information, and notify all third parties to "
            "whom the business has sold or shared such personal information to delete the "
            "consumer's personal information. Exceptions include when retention is "
            "necessary to complete a transaction, for security purposes, to comply with "
            "a legal obligation, or for internal uses reasonably aligned with consumer "
            "expectations."
        ),
        "violation_criteria": [
            "Refusing to delete personal information upon consumer request",
            "Not providing a mechanism for consumers to request deletion",
            "Not disclosing the right to deletion",
            "Failing to notify service providers to delete data",
            "Failing to notify third parties to delete data",
            "Ignoring deletion requests",
            "Making deletion unreasonably difficult",
            "Retaining data after a valid deletion request without a valid exception",
        ],
        "keywords": [
            "delete", "deletion", "erase", "remove", "right to delete",
            "data removal", "forget", "right to be forgotten", "retain",
            "refuse to delete", "ignore deletion", "keep data",
        ],
        "page_range": None,
    },
    "Section 1798.106": {
        "title": "Consumers' Right to Correct Inaccurate Personal Information",
        "text": (
            "Section 1798.106 provides that a consumer shall have the right to request "
            "that a business correct inaccurate personal information that it maintains "
            "about the consumer. A business that collects personal information about "
            "consumers shall disclose the consumer's right to request correction of "
            "inaccurate personal information. A business that receives a verifiable "
            "consumer request to correct inaccurate personal information shall use "
            "commercially reasonable efforts to correct the inaccurate personal "
            "information as directed by the consumer."
        ),
        "violation_criteria": [
            "Refusing to correct inaccurate personal information",
            "Not providing a mechanism for consumers to request correction",
            "Not disclosing the right to correction",
            "Ignoring correction requests",
            "Maintaining known inaccurate information about consumers",
        ],
        "keywords": [
            "correct", "correction", "inaccurate", "wrong", "error",
            "fix", "update", "amend", "modify", "accurate",
        ],
        "page_range": None,
    },
    "Section 1798.110": {
        "title": "Consumers' Right to Know What Personal Information is Being Collected",
        "text": (
            "Section 1798.110 provides that a consumer shall have the right to request "
            "that a business that collects personal information about the consumer "
            "disclose to the consumer the following: the categories of personal "
            "information it has collected about that consumer; the categories of sources "
            "from which the personal information is collected; the business or commercial "
            "purpose for collecting, selling, or sharing personal information; the "
            "categories of third parties to whom the business discloses personal "
            "information; and the specific pieces of personal information it has "
            "collected about that consumer."
        ),
        "violation_criteria": [
            "Refusing to disclose what personal information has been collected",
            "Not disclosing categories of personal information collected",
            "Not disclosing sources of personal information",
            "Not disclosing purposes of collection",
            "Not disclosing third parties to whom data is disclosed",
            "Not providing specific pieces of information upon request",
            "Ignoring consumer requests to know what data is collected",
        ],
        "keywords": [
            "know", "access", "disclose", "disclosure", "categories",
            "sources", "third parties", "request information", "transparency",
            "right to know", "what data", "what information",
        ],
        "page_range": None,
    },
    "Section 1798.115": {
        "title": "Consumers' Right to Know What Personal Information is Sold or Shared",
        "text": (
            "Section 1798.115 provides that a consumer shall have the right to request "
            "that a business that sells or shares the consumer's personal information "
            "disclose to the consumer: the categories of personal information that the "
            "business collected about the consumer; the categories of personal "
            "information that the business sold or shared about the consumer and the "
            "categories of third parties to whom the personal information was sold or "
            "shared; and the categories of personal information that the business "
            "disclosed about the consumer for a business purpose and the categories of "
            "persons to whom it was disclosed for a business purpose."
        ),
        "violation_criteria": [
            "Not disclosing what personal information is sold or shared",
            "Not disclosing categories of third parties data is sold or shared with",
            "Not disclosing categories of information disclosed for business purposes",
            "Refusing to respond to consumer requests about sold/shared data",
            "Hiding the fact that data is being sold or shared",
        ],
        "keywords": [
            "sell", "share", "sold", "shared", "third party", "disclose",
            "selling data", "sharing data", "who receives", "data broker",
            "ad network", "advertiser",
        ],
        "page_range": None,
    },
    "Section 1798.120": {
        "title": "Consumers' Right to Opt-Out of Sale or Sharing of Personal Information",
        "text": (
            "Section 1798.120 provides that a consumer shall have the right, at any "
            "time, to direct a business that sells or shares personal information about "
            "the consumer to third parties not to sell or share the consumer's personal "
            "information. This right may be referred to as the right to opt-out of the "
            "sale or sharing of personal information. A business that sells or shares "
            "consumers' personal information shall provide notice to consumers that their "
            "information may be sold or shared and that consumers have the right to "
            "opt out. A business shall not sell or share the personal information of "
            "consumers if the business has actual knowledge that the consumer is less "
            "than 16 years of age, unless the consumer, in the case of consumers at "
            "least 13 years of age and less than 16 years of age, or the consumer's "
            "parent or guardian, in the case of consumers who are less than 13 years "
            "of age, has affirmatively authorized the sale or sharing of the consumer's "
            "personal information. A business that has received direction from a "
            "consumer not to sell or share the consumer's personal information shall "
            "wait for at least 12 months before requesting that the consumer authorize "
            "the sale of the consumer's personal information."
        ),
        "violation_criteria": [
            "Selling personal information without providing opt-out option",
            "Sharing personal information without providing opt-out option",
            "Selling data of consumers under 16 without affirmative authorization",
            "Selling data of consumers under 13 without parental consent",
            "Not providing a 'Do Not Sell or Share' link or mechanism",
            "Ignoring consumer opt-out requests",
            "Selling data after consumer has opted out",
            "Asking consumer to opt back in before 12 months have passed",
            "Selling personal information to third parties without consumer knowledge",
            "Not providing notice that information may be sold or shared",
        ],
        "keywords": [
            "sell", "sale", "opt out", "opt-out", "share", "sharing",
            "third party", "ad network", "data broker", "minor", "child",
            "children", "under 16", "under 13", "teenager", "parental consent",
            "do not sell", "without consent", "without permission",
            "without knowledge", "without notifying",
        ],
        "page_range": None,
    },
    "Section 1798.121": {
        "title": "Consumers' Right to Limit Use and Disclosure of Sensitive Personal Information",
        "text": (
            "Section 1798.121 provides that a consumer shall have the right, at any "
            "time, to direct a business that collects sensitive personal information "
            "about the consumer to limit its use of the consumer's sensitive personal "
            "information to that use which is necessary to perform the services or "
            "provide the goods reasonably expected by an average consumer who requests "
            "those goods or services. Sensitive personal information includes social "
            "security numbers, driver's license numbers, financial account information, "
            "precise geolocation, racial or ethnic origin, religious beliefs, union "
            "membership, contents of mail/email/text messages, genetic data, biometric "
            "information, health information, and sex life or sexual orientation data. "
            "A business shall provide a link titled 'Limit the Use of My Sensitive "
            "Personal Information' on its website or app."
        ),
        "violation_criteria": [
            "Using sensitive personal information beyond what is necessary for the service",
            "Not providing mechanism to limit use of sensitive personal information",
            "Processing sensitive data without consumer consent or notice",
            "Sharing sensitive personal information unnecessarily",
            "Using biometric, health, financial, or genetic data for undisclosed purposes",
            "Not providing 'Limit the Use of My Sensitive Personal Information' link",
            "Collecting precise geolocation without proper disclosure",
            "Using racial, ethnic, religious, or sexual orientation data improperly",
        ],
        "keywords": [
            "sensitive", "biometric", "genetic", "health", "geolocation",
            "social security", "driver's license", "financial", "racial",
            "ethnic", "religious", "sexual orientation", "sex life",
            "union membership", "precise location", "SSN", "fingerprint",
            "face recognition", "facial recognition", "DNA",
        ],
        "page_range": None,
    },
    "Section 1798.125": {
        "title": "Consumers' Right of No Retaliation / Non-Discrimination",
        "text": (
            "Section 1798.125 prohibits a business from discriminating against a "
            "consumer because the consumer exercised any of the consumer's rights under "
            "the CCPA, including, but not limited to: denying goods or services to the "
            "consumer; charging different prices or rates for goods or services, "
            "including through the use of discounts or other benefits or imposing "
            "penalties; providing a different level or quality of goods or services to "
            "the consumer; or suggesting that the consumer will receive a different "
            "price or rate for goods or services or a different level or quality of "
            "goods or services. A business may offer financial incentives, including "
            "payments to consumers as compensation, for the collection of personal "
            "information, the sale or sharing of personal information, or the retention "
            "of personal information. A business may also offer a different price, rate, "
            "level, or quality of goods or services to the consumer if that price or "
            "difference is reasonably related to the value provided to the business by "
            "the consumer's data."
        ),
        "violation_criteria": [
            "Denying goods or services to consumers who exercise CCPA rights",
            "Charging different prices to consumers who exercise CCPA rights",
            "Providing different quality of service based on CCPA right exercise",
            "Retaliating against consumers who opt out of data sales",
            "Penalizing consumers for requesting deletion or correction",
            "Discriminating against consumers who exercise privacy rights",
            "Threatening consumers with degraded service for exercising rights",
        ],
        "keywords": [
            "discriminat", "retaliat", "penaliz", "deny", "charge more",
            "different price", "different quality", "punish", "degrade",
            "service quality", "price increase", "penalty", "incentive",
            "loyalty program", "financial incentive",
        ],
        "page_range": None,
    },
    "Section 1798.130": {
        "title": "Notice, Disclosure, Correction, and Deletion Requirements",
        "text": (
            "Section 1798.130 sets out how businesses must handle consumer requests. "
            "A business must make available to consumers two or more designated methods "
            "for submitting requests for information, including at a minimum a toll-free "
            "telephone number. A business must respond to a verifiable consumer request "
            "within 45 calendar days of receiving the request. The time period may be "
            "extended once by an additional 45 days when reasonably necessary, provided "
            "the consumer is informed. The response shall cover the 12-month period "
            "preceding the business's receipt of the verifiable consumer request. "
            "Disclosures shall be delivered free of charge and the information may be "
            "delivered in a portable and readily usable format. A business must disclose "
            "and deliver the required information to a consumer free of charge within "
            "45 days. A business shall not require a consumer to create an account to "
            "make a request. A business shall maintain records of consumer requests for "
            "at least 24 months."
        ),
        "violation_criteria": [
            "Not responding to consumer requests within 45 days",
            "Not providing designated methods for submitting requests",
            "Not providing a toll-free telephone number for requests",
            "Charging consumers for disclosing their personal information",
            "Requiring account creation to submit a request",
            "Not maintaining records of consumer requests for 24 months",
            "Not providing information in a portable/usable format",
            "Not providing at least two methods for submitting requests",
        ],
        "keywords": [
            "respond", "response time", "45 days", "request", "toll-free",
            "telephone", "method", "submit", "account", "records",
            "portable", "format", "free of charge", "timeline", "deadline",
            "verifiable", "verification",
        ],
        "page_range": None,
    },
    "Section 1798.135": {
        "title": "Methods of Limiting Sale, Sharing, and Use of Personal Information",
        "text": (
            "Section 1798.135 requires a business that sells or shares consumers' "
            "personal information or uses or discloses consumers' sensitive personal "
            "information to provide a clear and conspicuous link on the business's "
            "internet homepage titled 'Do Not Sell or Share My Personal Information' "
            "that enables a consumer to opt out of the sale or sharing of the consumer's "
            "personal information. A business shall also provide a clear and conspicuous "
            "link on the business's internet homepage titled 'Limit the Use of My "
            "Sensitive Personal Information' if applicable. A business may provide a "
            "single, clearly-labeled link on the business's internet homepage, in lieu "
            "of the two links, if such link easily allows a consumer to opt out of the "
            "sale or sharing and to limit the use of sensitive personal information. "
            "A business shall respect the consumer's decision to opt out for at least "
            "12 months. A business shall process opt-out preference signals as valid "
            "requests to opt out."
        ),
        "violation_criteria": [
            "Not providing a 'Do Not Sell or Share' link on website",
            "Not providing a 'Limit the Use of My Sensitive Personal Information' link",
            "Making the opt-out link unclear or hidden",
            "Not honoring opt-out preference signals (e.g., GPC)",
            "Requiring consumers to create account to opt out",
            "Not processing opt-out requests properly",
            "Making the opt-out process unnecessarily difficult",
        ],
        "keywords": [
            "do not sell", "opt out link", "homepage", "opt-out signal",
            "GPC", "global privacy control", "preference signal", "link",
            "website", "conspicuous", "clear", "easy", "mechanism",
        ],
        "page_range": None,
    },
    "Section 1798.150": {
        "title": "Personal Information Security Breaches",
        "text": (
            "Section 1798.150 provides that any consumer whose nonencrypted and "
            "nonredacted personal information is subject to an unauthorized access and "
            "exfiltration, theft, or disclosure as a result of the business's violation "
            "of the duty to implement and maintain reasonable security procedures and "
            "practices appropriate to the nature of the information to protect the "
            "personal information may institute a civil action for any of the following: "
            "to recover damages in an amount not less than one hundred dollars ($100) "
            "and not greater than seven hundred and fifty dollars ($750) per consumer per "
            "incident or actual damages, whichever is greater; injunctive or declaratory "
            "relief; or any other relief the court deems proper. Prior to initiating any "
            "action, a consumer shall provide the business 30 days' written notice "
            "identifying the specific provisions that have been or are being violated."
        ),
        "violation_criteria": [
            "Failing to implement reasonable security procedures",
            "Failing to maintain reasonable security practices",
            "Storing personal information without encryption",
            "Storing personal information without redaction where appropriate",
            "Data breach due to inadequate security measures",
            "Not protecting personal information from unauthorized access",
            "Failing to implement data security safeguards",
            "Negligent handling of personal information leading to exposure",
        ],
        "keywords": [
            "security", "breach", "hack", "unauthorized access", "encrypt",
            "encryption", "protect", "safeguard", "leak", "exposure",
            "theft", "stolen", "compromise", "vulnerability", "plain text",
            "unencrypted", "data breach", "cybersecurity", "negligent",
        ],
        "page_range": None,
    },
}

# ── Section lookup helper ────────────────────────────────────
def get_section_text(section_id: str) -> str:
    """Return the full text for a given section ID."""
    sec = CCPA_SECTIONS.get(section_id)
    if sec:
        return f"{section_id} — {sec['title']}\n{sec['text']}"
    return ""


def get_all_sections_text() -> str:
    """Return all sections concatenated for context."""
    parts = []
    for sid, sec in CCPA_SECTIONS.items():
        parts.append(
            f"{sid} — {sec['title']}\n"
            f"{sec['text']}\n"
            f"Violation criteria: {'; '.join(sec['violation_criteria'])}"
        )
    return "\n\n".join(parts)


def get_section_ids() -> list:
    """Return list of all section IDs."""
    return list(CCPA_SECTIONS.keys())