"""
build_clusters.py
Reads explicit_network_layout.json and the readable names CSV.
Applies a hand-built CELEX → cluster mapping (primary), then a keyword-based
auto-classifier for unmapped laws.  Writes proposed_clustering.csv for human
review BEFORE we touch the JSON.
"""
import csv
import json
import re
from pathlib import Path

BASE       = Path(__file__).parent
LAYOUT     = BASE / "explicit_network_layout.json"
NAMES_CSV  = Path(r"C:\ADS MASTER\THESIS PROJECT\Final\Data\260512-laws-celex-and-human-readable-names.csv")
OUT        = BASE / "proposed_clustering_v2.csv"

CLUSTER_NAMES = {
    0:  "Data Protection & Privacy",
    1:  "AI, Data Economy & Digital Identity",
    2:  "Platform Regulation & Digital Markets",
    3:  "Digital Content, Media & Copyright",
    4:  "Cybersecurity & Operational Resilience",
    5:  "Financial Markets, Investment & Capital",
    6:  "Banking, Payments, Insurance & AML",
    7:  "Consumer Rights & Internal Market",
    8:  "Intellectual Property",
    9:  "Criminal Justice & Fundamental Rights",
    10: "Migration, Asylum & Border Management",
    11: "Transport & Mobility",
    12: "Environment, Energy & Climate",
    13: "Health, Medicines & Life Sciences",
    14: "Agriculture, Fisheries & Rural Development",
    15: "EU Institutional Framework & Funding",
    19: "Other",
}

# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY HAND-BUILT CELEX → CLUSTER MAP  (~415 laws)
# ─────────────────────────────────────────────────────────────────────────────
CLUSTER_MAP = {
    # ─── 0. Data Protection & Privacy ────────────────────────────────
    "32016R0679": 0,  # GDPR
    "32002L0058": 0,  # ePrivacy
    "32016L0680": 0,  # LED
    "32008F0977": 0,  # FD 977 (police data, repealed by LED)
    "32018R1725": 0,  # EUDPR
    "32025R2518": 0,  # GDPR Enforcement Procedural Rules
    "32025R0327": 0,  # European Health Data Space

    # ─── 1. AI, Data Economy & Digital Identity ──────────────────────
    "32024R1689": 1,  # AI Act
    "32023R2854": 1,  # Data Act
    "32022R0868": 1,  # DGA
    "32018R1807": 1,  # Free Flow Non-Personal Data
    "32014R0910": 1,  # eIDAS 1
    "32024R1183": 1,  # eIDAS 2
    "32023R1781": 1,  # Chips Act
    "32021R0694": 1,  # Digital Europe Programme
    "32024R0903": 1,  # Interoperable Europe Act
    "32022R0850": 1,  # e-CODEX
    "32023R2844": 1,  # Digitalisation Cross-Border Judicial
    "32021R2282": 1,  # HTA

    # ─── 2. Platform Regulation & Digital Markets ────────────────────
    "32022R2065": 2,  # DSA
    "32022R1925": 2,  # DMA
    "32000L0031": 2,  # eCommerce
    "32019L0770": 2,  # Digital Content Directive
    "32021R0784": 2,  # Terrorist Content Regulation
    "32024R0900": 2,  # TTPAR Political Advertising
    "32018L1972": 2,  # EECC
    "32015R2120": 2,  # Net Neutrality
    "32022R0612": 2,  # Roaming
    "32024R1309": 2,  # Gigabit Infrastructure Act
    "32021R1232": 2,  # Temporary CSAM
    "32019L1024": 2,  # Open Data Directive
    "32008L0006": 2,  # Postal Services Internal Market

    # ─── 3. Digital Content, Media & Copyright ───────────────────────
    "32018L1808": 3,  # AVMSD
    "32024R1083": 3,  # EMFA
    "32019L0790": 3,  # DSM Copyright
    "32001L0029": 3,  # Copyright InfoSoc
    "32014L0026": 3,  # Collective Rights Management
    "32017L1564": 3,  # Marrakesh Implementation
    "32017R1563": 3,  # Marrakesh Cross-Border Exchange
    "32017R1128": 3,  # Cross-Border Portability

    # ─── 4. Cybersecurity & Operational Resilience ───────────────────
    "32022L2555": 4,  # NIS2
    "32019R0881": 4,  # Cybersecurity Act
    "32022R2554": 4,  # DORA
    "32022L2556": 4,  # DORA Amending Directive
    "32024R2847": 4,  # Cyber Resilience Act
    "32025R0038": 4,  # Cyber Solidarity Act
    "32023R2841": 4,  # EU-INST Cybersecurity Regulation
    "32022L2557": 4,  # CER Directive
    "32021R0887": 4,  # ECCC
    "32014L0053": 4,  # Radio Equipment Directive

    # ─── 5. Financial Markets, Investment & Capital ──────────────────
    "32014L0065": 5,  # MiFID II
    "32016L1034": 5,  # MiFID II amending
    "32014R0600": 5,  # MiFIR
    "32012R0648": 5,  # EMIR
    "32013R0575": 5,  # CRR
    "32014R0596": 5,  # MAR
    "32014L0057": 5,  # MAD Criminal
    "32012R0236": 5,  # Short Selling
    "32014R0909": 5,  # CSDR
    "32016R1011": 5,  # Benchmarks
    "32015R2365": 5,  # SFTR
    "32017R2402": 5,  # Securitisation
    "32014R1286": 5,  # PRIIPs
    "32017R1129": 5,  # Prospectus
    "32017R1131": 5,  # MMF
    "32011L0061": 5,  # AIFMD
    "32014L0091": 5,  # UCITS V
    "32013R0345": 5,  # EuVECA
    "32013R0346": 5,  # EuSEF
    "32015R0760": 5,  # ELTIF
    "32023R0606": 5,  # ELTIF 2
    "32009R1060": 5,  # CRAs
    "32011R0513": 5,  # CRA amending
    "32013R0462": 5,  # CRA amending 2
    "32019L2034": 5,  # Investment Firms Directive
    "32010R1093": 5,  # EBA
    "32010R1094": 5,  # EIOPA
    "32010R1095": 5,  # ESMA
    "32021R0023": 5,  # CCP Recovery
    "32024R2987": 5,  # Clearing Markets Efficiency
    "32022R0858": 5,  # DLT Pilot
    "32023R1114": 5,  # MiCA
    "32023R2631": 5,  # Green Bonds
    "32024R3005": 5,  # ESG Ratings
    "32024R2809": 5,  # Listing Act
    "32004L0109": 5,  # Transparency Directive
    "32013L0050": 5,  # Transparency Directive Amend
    "32017L0828": 5,  # Shareholder Rights II
    "32014L0095": 5,  # NFRD
    "32011R1227": 5,  # REMIT
    "32024R1106": 5,  # REMIT II
    "32023R2859": 5,  # European Single Access Point
    "32016R0867": 5,  # AnaCredit
    "32006L0043": 5,  # Statutory Audit Directive
    "32014L0056": 5,  # Statutory Audit Amending Directive
    "32014R0537": 5,  # Statutory Audit PIE
    "32017L1132": 5,  # Company Law Directive
    "32019L1151": 5,  # Digital Company Law
    "32013R1409": 5,  # ECB payments stats
    "32021L0338": 5,  # MiFID II quick fix
    "32016R1033": 5,  # MiFIR delegated/amend
    "32017R2396": 5,  # CRR transitional amend
    "32023R2845": 5,  # CSDR amending
    "32025L0025": 5,  # Company law amend
    "32023L2864": 5,  # Listing Act amending Dir
    "32023L2661": 5,  # Cap ratios amend
    "32024R0791": 5,  # CRR amend

    # ─── 6. Banking, Payments, Insurance & AML ───────────────────────
    "32013L0036": 6,  # CRD IV
    "32019L0878": 6,  # CRD V
    "32024L1619": 6,  # CRD VI
    "32014L0059": 6,  # BRRD
    "32014R0806": 6,  # SRM
    "32014L0049": 6,  # DGS
    "32015L2366": 6,  # PSD2
    "32014L0092": 6,  # Payment Accounts
    "32024R0886": 6,  # Instant Payments
    "32012R0260": 6,  # SEPA
    "32018L0843": 6,  # AML5
    "32024L1640": 6,  # AML6
    "32018L1673": 6,  # AML Criminal
    "32024R1620": 6,  # AMLAR
    "32024R1624": 6,  # AMLR
    "32023R1113": 6,  # Transfer of Funds
    "32008L0048": 6,  # Consumer Credit
    "32023L2225": 6,  # Consumer Credit new
    "32002L0065": 6,  # Distance Marketing Financial Services
    "32016L0097": 6,  # Insurance Distribution
    "32019L2177": 6,  # Solvency II
    "32025L0001": 6,  # Insurance Recovery
    "32025L0002": 6,  # Solvency II Amendment
    "32016L2341": 6,  # IORP II
    "32019L2162": 6,  # Covered Bonds
    "32021L2167": 6,  # Credit Servicers
    "32014R0468": 6,  # SSM Framework
    "32010L0045": 6,  # VAT e-invoicing amend
    "32022L0228": 6,  # VAT amend
    "32024L1226": 6,  # VAT amend
    "32021L2118": 6,  # Motor Insurance amend
    "32024R3011": 6,  # banking amend
    "32024R0897": 6,  # financial assistance amend

    # ─── 7. Consumer Rights & Internal Market ────────────────────────
    "32011L0083": 7,  # Consumer Rights
    "32019L2161": 7,  # Omnibus
    "32020L1828": 7,  # Class Action
    "32013R0524": 7,  # Consumer ODR
    "32006L0123": 7,  # Services Directive
    "32005L0036": 7,  # Professional Qualifications
    "32012R1024": 7,  # IMI
    "32014L0023": 7,  # Procurement Concessions
    "32014L0024": 7,  # Procurement Classical
    "32014L0025": 7,  # Procurement Utilities
    "32014L0055": 7,  # Electronic Invoicing
    "32019L0633": 7,  # Unfair Trading Practices
    "32024L2853": 7,  # Product Liability Revision
    "32023R0988": 7,  # General Product Safety
    "32019R1020": 7,  # MSR
    "32008R0765": 7,  # Accreditation+Market Surveillance
    "32024R2747": 7,  # Internal Market Emergency
    "32021R0690": 7,  # Single Market Programme
    "32008L0122": 7,  # Timeshare
    "32007R0861": 7,  # Small Claims
    "32007R0864": 7,  # Rome II
    "32007R1393": 7,  # Service of Documents
    "32012R1215": 7,  # Brussels I Recast
    "32014R0655": 7,  # European Account Preservation Order
    "32015R0848": 7,  # Insolvency Regulation
    "32016R1191": 7,  # Public Documents
    "32017R2394": 7,  # CPC Cooperation
    "32004R2006": 7,  # CPC
    "32024L2831": 7,  # Platform Work
    "32023L0970": 7,  # Pay Transparency
    "32022L2381": 7,  # Women on Boards
    "32022R2560": 7,  # Foreign Subsidies
    "32016R0589": 7,  # EURES
    "32013R0606": 7,  # Mutual recognition civil protection
    "32024L1654": 7,  # Insolvency amend
    "32024L1500": 7,  # Gender Equality Bodies
    "32024L1385": 7,  # Combating Gender-Based Violence
    "32025R2509": 7,  # Toys Safety
    "32024R3110": 7,  # Construction Products
    "32023R1230": 7,  # Machinery
    "32024R1735": 7,  # Net-Zero Industry Act
    "32009R0987": 7,  # Social Security Coordination Implementing
    "32021R0691": 7,  # EGF Regulation

    # ─── 8. Intellectual Property ────────────────────────────────────
    "32004L0048": 8,  # IPRED
    "32015L2436": 8,  # Trade Marks Directive
    "32017R1001": 8,  # EU Trade Mark Regulation
    "32013R0608": 8,  # Customs IP Enforcement
    "32016L0943": 8,  # Trade Secrets
    "32024L2823": 8,  # Designs Directive
    "32024R2822": 8,  # Community Designs
    "32006R0816": 8,  # Compulsory Licensing Export
    "32012R0386": 8,  # OHIM/EUIPO
    "32023R2411": 8,  # Craft GI
    "32024R1143": 8,  # Geographical Indications

    # ─── 9. Criminal Justice & Fundamental Rights ────────────────────
    "12007P":     9,  # Charter
    "32002F0475": 9,  # Terrorism FD
    "32005D0671": 9,  # Terrorism info exchange
    "32002F0465": 9,  # Joint Investigation Teams
    "32002D0348": 9,  # Football security
    "32008D0615": 9,  # Prüm Decision
    "32008D0616": 9,  # Prüm Implementation
    "32008D0633": 9,  # VIS-police access
    "32009F0315": 9,  # Criminal records exchange
    "32008F0909": 9,  # Transfer of sentences
    "32008F0947": 9,  # Suspended sentences
    "32009F0829": 9,  # Supervision Order
    "32009F0905": 9,  # Forensic accreditation
    "32012L0013": 9,  # Right to Information criminal
    "32014L0041": 9,  # EIO Directive
    "32011L0099": 9,  # European Protection Order
    "32017L1371": 9,  # PIF Directive
    "32022R0838": 9,  # Eurojust War Crimes
    "32023R2131": 9,  # Digital Info Exchange Terrorism
    "32016R0794": 9,  # Europol
    "32022R0991": 9,  # Europol amended
    "32021L0555": 9,  # Firearms
    "32025R0041": 9,  # EU Firearms
    "32012R0258": 9,  # Firearms export
    "32024L1260": 9,  # Asset Recovery
    "32024L1203": 9,  # Environmental Crime
    "32019L1937": 9,  # Whistleblowing
    "32020R2223": 9,  # OLAF
    "32024L1069": 9,  # Anti-SLAPP
    "32023L1544": 9,  # e-Evidence Directive
    "32023R1543": 9,  # e-Evidence Regulation
    "32024R0982": 9,  # Prüm II
    "32024L2810": 9,  # Criminal amend
    "32023R2869": 9,  # Criminal records amend
    "32016R2030": 9,  # OLAF amend
    "32023R2667": 9,  # Sanctions amend
    "32014R1141": 9,  # EU Political Parties

    # ─── 10. Migration, Asylum & Border Management ───────────────────
    "32013R0604": 10, # Dublin III
    "32013L0032": 10, # Asylum Procedures Directive
    "32013R0603": 10, # Eurodac
    "32024R1358": 10, # Eurodac new
    "32024R1348": 10, # Asylum Procedures Regulation
    "32024R1351": 10, # AMR
    "32024R1356": 10, # Screening
    "32024R1350": 10, # Resettlement
    "32024R1352": 10, # Migration amend
    "32021R2303": 10, # EU Asylum Agency
    "32021R1147": 10, # AMIF
    "32016R0399": 10, # Schengen Borders Code
    "32013R0610": 10, # SBC amendment
    "32017R2225": 10, # EES-SBC amendment
    "32017R2226": 10, # EES
    "32009R0810": 10, # Visa Code
    "32008R0767": 10, # VIS
    "32021R1133": 10, # VIS Amending
    "32021R1134": 10, # VIS new
    "32021R1150": 10, # ETIAS Interoperability
    "32021R1151": 10, # ETIAS
    "32021R1152": 10, # ETIAS Interoperability
    "32014R0515": 10, # ISF Borders Visa
    "32014R0656": 10, # Sea Borders
    "32025R0012": 10, # API Regulation
    "32025R0013": 10, # API-PNR Router
    "32021R1148": 10, # Border Management Fund
    "32021R1149": 10, # Internal Security Fund
    "32024R1717": 10, # Schengen amend
    "32005R1160": 10, # SIS amend
    "32014R0513": 10, # External Borders Fund amend
    "32014R0514": 10, # Migration Fund general provisions
    "32022R1190": 10, # SIS Information Alerts
    "32021R1077": 10, # Customs Control Equipment
    "32023R0969": 10, # Refugees amend
    "32023R2685": 10, # Schengen visa amend

    # ─── 11. Transport & Mobility ────────────────────────────────────
    "32007L0059": 11, # Train Drivers
    "32010R0996": 11, # Civil Aviation Accident Investigation
    "32014R0376": 11, # Civil Aviation Occurrence Reporting
    "32010R1177": 11, # Maritime Passenger Rights
    "32011R0181": 11, # Bus and Coach
    "32006R1107": 11, # Air Passenger Rights Disabled
    "32021R0782": 11, # Rail Passengers Rights
    "32002R1406": 11, # EMSA
    "32011R1286": 11, # Marine casualty
    "32009R0392": 11, # Passenger liability sea
    "32015L0413": 11, # Cross-Border Traffic Offences
    "32015R0758": 11, # eCall
    "32019L0520": 11, # EETS
    "32014L0047": 11, # Roadside inspection
    "32014L0046": 11, # Vehicle registration
    "32005L0044": 11, # RIS
    "32017L2397": 11, # Inland Navigation
    "32014R0165": 11, # Tachographs
    "32023R1804": 11, # Alternative Fuels Infrastructure
    "32024R1679": 11, # TEN-T
    "32024R1257": 11, # Euro 7
    "32016R0796": 11, # ERA
    "32017R0352": 11, # Port Services
    "32015R0757": 11, # MRV Maritime
    "32022L2561": 11, # Driver Qualification
    "32016L1629": 11, # Inland waterways
    "32018L0645": 11, # Driving licences amend
    "32010L0040": 11, # ITS Directive
    "32009L0018": 11, # Marine accident investigation
    "32016L2370": 11, # Rail
    "32017L2109": 11, # Passenger registration on ships
    "32022L0738": 11, # Roadworthiness amend
    "32022L0993": 11, # Seafarers training
    "32017R2403": 11, # External Fishing Fleets

    # ─── 12. Environment, Energy & Climate ───────────────────────────
    "32023R0956": 12, # CBAM
    "32018R1999": 12, # Energy Union Governance
    "32019L0944": 12, # Electricity Directive
    "32024R1789": 12, # Gas/Hydrogen Markets
    "32024L1788": 12, # Gas/Hydrogen Markets Dir
    "32024R1787": 12, # Methane
    "32024L1275": 12, # EPBD
    "32024R1781": 12, # Ecodesign
    "32017R1369": 12, # Energy Labelling
    "32024R0573": 12, # F-Gas
    "32024R0590": 12, # Ozone
    "32024R1991": 12, # Nature Restoration
    "32023R1115": 12, # Deforestation-Free
    "32021R1056": 12, # Just Transition
    "32023R0955": 12, # Social Climate Fund
    "32023R0435": 12, # REPowerEU
    "32024R3012": 12, # Carbon Removals
    "32024L3019": 12, # Urban Wastewater
    "32024R1157": 12, # Waste Shipments
    "32025R0040": 12, # Packaging
    "32022R0869": 12, # TEN-E
    "32024L1711": 12, # Electricity amend
    "32024L1712": 12, # Electricity amend
    "32024R1747": 12, # Electricity market reform

    # ─── 13. Health, Medicines & Life Sciences ───────────────────────
    "32004L0023": 13, # Human Tissues
    "32004R0726": 13, # EMA
    "32004R0851": 13, # ECDC
    "32010L0084": 13, # Pharmacovigilance
    "32010R1235": 13, # Pharmacovigilance amend
    "32011L0062": 13, # Falsified Medicines
    "32011L0024": 13, # Cross-border Healthcare
    "32014R0536": 13, # CTR
    "32017R0745": 13, # MDR
    "32017R0746": 13, # IVDR
    "32021R0522": 13, # EU4Health
    "32022R0123": 13, # HERA/EMA mandate
    "32022R2370": 13, # ECDC amend
    "32022R2371": 13, # Cross-Border Health Threats
    "32024R1938": 13, # Substances Human Origin
    "32014L0040": 13, # Tobacco Products
    "32023R1322": 13, # EUDA
    "32016R0793": 13, # Avoid diversion of medicines
    "32022R1034": 13, # COVID cert citizens
    "32022R1035": 13, # COVID cert third country

    # ─── 14. Agriculture, Fisheries & Rural Development ──────────────
    "32013R1380": 14, # CFP
    "32013R1379": 14, # CMO Fisheries
    "32023R2124": 14, # GFCM Fishing
    "32023R2842": 14, # Fisheries Control
    "32013R1305": 14, # Rural Development
    "32013R1306": 14, # CAP Horizontal
    "32013R1308": 14, # Single CMO
    "32021R2115": 14, # CAP Strategic Plans
    "32021R2116": 14, # CAP Financing
    "32021R2117": 14, # CAP amending
    "32023R2674": 14, # FSDN
    "32017R0625": 14, # Official Controls
    "32016R1012": 14, # Animal Breeding
    "32021R1139": 14, # EMFAF
    "32014R0508": 14, # EMFF
    "32017R1004": 14, # Fisheries Data Collection
    "32014R0223": 14, # FEAD (food aid)
    "32023R0675": 14, # CAP amend
    "32023R1092": 14, # CAP horizontal amend
    "32023R2053": 14, # Fisheries control amend
    "32022R2056": 14, # NAFO amend
    "32022R2343": 14, # GFCM amend
    "32022R2379": 14, # Agri statistics
    "32024R1244": 14, # CAP amend
    "32024R1307": 14, # CAP amend
    "32024R1028": 14, # STR (sustainable food)

    # ─── 15. EU Institutional Framework & Funding ────────────────────
    "32003R1882": 15, # Comitology Adaptation
    "32009R0596": 15, # Comitology amend
    "32013R1023": 15, # Staff Regulations
    "32021R0241": 15, # RRF
    "32021R0523": 15, # InvestEU
    "32021R0695": 15, # Horizon Europe
    "32021R0696": 15, # Space Programme
    "32021R0697": 15, # European Defence Fund
    "32021R0817": 15, # Erasmus+
    "32021R0818": 15, # Creative Europe
    "32021R1057": 15, # ESF+
    "32021R1060": 15, # Common Provisions
    "32021R1153": 15, # CEF
    "32021R1755": 15, # Brexit Adjustment Reserve
    "32021R0947": 15, # Global Europe
    "32021R1529": 15, # IPA III
    "32024R0792": 15, # Ukraine Facility
    "32024R2509": 15, # EU Financial Regulation
    "32006R1922": 15, # EIGE
    "32008R1339": 15, # ETF
    "32021R1163": 15, # European Ombudsman Statute
    "32021R0240": 15, # TSI
    "32021R0444": 15, # Customs Programme
    "32021R0692": 15, # CERV Programme
    "32021R0819": 15, # Erasmus amend
    "32021R1229": 15, # Public Sector Loan Facility (JT)
    "32023R0588": 15, # Union Secure Connectivity
    "32021R0821": 15, # Dual-Use Export Control
    "32023R2675": 15, # Anti-Coercion Instrument
    "32009R0223": 15, # European Statistics
    "32007L0002": 15, # INSPIRE
    "32007R1445": 15, # Statistics income/living
    "32008R0452": 15, # Statistics plant protection
    "32023R2833": 15, # Statistics tourism
    "32024R0568": 15, # Budget corrections
    "32024R1449": 15, # Sanctions extension
    "32024R3015": 15, # Forced Labour Regulation
    "32025R1106": 15, # Security Action for Europe

    # ─── Hand-classified residuals (56 laws) ────────────────────────
    # 15: EU Institutional / oldest foundational instruments
    "31994L0022": 12, # Hydrocarbons Licensing Directive — energy sector
    "31958R0001": 15, # EEC Reg 58/1 — official languages of EEC institutions
    "31977L0799": 15, # Directive 77/799 — mutual assistance direct tax
    "31988L0361": 5,  # Directive 88/361 — capital movement liberalisation
    "32006L0096": 15, # Directive 2006/96 — accession adaptation
    "32011R1179": 15, # Reg 1179/2011 — European Citizens' Initiative technical reg
    "32012L0017": 15, # Directive 2012/17 — business registers interconnection
    "32013L0025": 15, # Directive 2013/25 — accession adaptation (Croatia)
    "32014R0463": 15, # Reg 463/2014 — cohesion fund IT system
    "32014R1312": 15, # Reg 1312/2014 — INSPIRE implementing
    "32015R0341": 15, # Reg 2015/341 — FEAD fund implementing
    "32020R0559": 15, # Reg 2020/559 — FEAD COVID emergency
    "32021R0177": 15, # Reg 2021/177 — social fund implementing
    "32021R0629": 15, # Reg 2021/629 — FEAD COVID
    "32024L1265": 15, # Directive 2024/1265 — EU fiscal/budget framework (stability pact)
    "32025R2088": 5,  # Reg 2025/2088 — financial market transparency/ESG

    # 13: Health / Food Safety
    "31980L0777": 13, # Directive 80/777 — natural mineral waters
    "31999L0004": 13, # Directive 1999/4 — coffee and chicory products
    "32009L0054": 13, # Directive 2009/54 — mineral water (recast)

    # 14: Agriculture / Fisheries
    "31999R0856": 14, # Reg 856/1999 — ACP fruit and vegetables development
    "32019R2074": 14, # Reg 2019/2074 — customs food inspection (official controls)

    # 7: Consumer Rights / Internal Market / Civil Justice
    "31989L0106": 7,  # Construction Products Directive 89/106
    "32000R1348": 7,  # Reg 1348/2000 — old service of documents (superseded)
    "32005R0603": 7,  # Reg 603/2005 — insolvency proceedings / civil procedure
    "32009R0662": 7,  # Reg 662/2009 — judicial cooperation civil matters
    "32013R0681": 7,  # Reg 681/2013 — toy safety implementing
    "32014L0079": 7,  # Directive 2014/79 — toy safety amend
    "32014R0542": 7,  # Reg 542/2014 — civil courts / patent court
    "32015R0281": 7,  # Reg 2015/281 — judicial cooperation civil matters (accession)
    "32015R1051": 7,  # Reg 2015/1051 — ODR consumer platform implementing
    "32016R1103": 7,  # Reg 2016/1103 — matrimonial property
    "32016R1104": 7,  # Reg 2016/1104 — registered partnerships property
    "32018R0946": 7,  # Reg 2018/946 — judicial cooperation civil matters implementing
    "32022R0423": 7,  # Reg 2022/423 — e-CODEX / e-justice civil cooperation
    "32024R1570": 7,  # Reg 2024/1570 — e-justice / judicial cooperation IT

    # 6: Banking / Payments / Insurance
    "32011L0090": 6,  # Directive 2011/90 — consumer credit cost calculation
    "32014R1163": 6,  # Reg 1163/2014 — ECB supervisory fees
    "32017R0867": 6,  # Reg 2017/867 — bank asset transfer / stabilisation
    "32017R1469": 6,  # Reg 2017/1469 — insurance information standardised document
    "32018L0411": 6,  # Directive 2018/411 — insurance implementing
    "32018R0541": 6,  # Reg 2018/541 — insurance distribution implementing
    "32021R1257": 6,  # Reg 2021/1257 — Solvency II sustainability
    "32022R1011": 6,  # Reg 2022/1011 — banking technical standard
    "32022R2036": 6,  # Reg 2022/2036 — bank resolution / subsidiary requirements
    "32023R1577": 6,  # Reg 2023/1577 — banking technical standard
    "32023R1578": 6,  # Reg 2023/1578 — banking technical standard

    # 8: Intellectual Property
    "32004L0048R(01)": 8,  # IPRED corrigendum

    # 5: Financial Markets / Capital
    "32016R1014": 5,  # Reg 2016/1014 — commodity derivatives / benchmark
    "32019R0819": 5,  # Reg 2019/819 — ESG/financial disclosure
    "32022R2117": 5,  # Reg 2022/2117 — crowdfunding technical standard
    "32024R0358": 5,  # Reg 2024/358 — crowdfunding implementing

    # 4: Cybersecurity / Operational Resilience
    "32023R1717": 4,  # Reg 2023/1717 — radio equipment standard (RED implementing)
    "32024R2690": 4,  # Reg 2024/2690 — NIS2 / cybersecurity risk management
    "32025R0037": 4,  # Reg 2025/37 — IT security services

    # 2: Platform Regulation / Digital Markets
    "32023R0138": 2,  # Reg 2023/138 — open data re-use portal (data.europa.eu)
    "32023R0444": 2,  # Reg 2023/444 — emergency comms (112 / telecom)
    "32019R2243": 2,  # Reg 2019/2243 — emergency comms telecom standard
}

# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD-BASED AUTO-CLASSIFIER
# Applied only when a CELEX is NOT in CLUSTER_MAP.
# Rules are evaluated top-to-bottom; first match wins.
# Each entry:  (regex_pattern, cluster_id)
# The haystack is a single lowercase string built from ALL metadata fields.
# ─────────────────────────────────────────────────────────────────────────────
KEYWORD_RULES = [

    # ── 0. Data Protection & Privacy ─────────────────────────────────
    (r"data protection directive|dpd|eudpr|data retention directive"
     r"|cookie directive|telecom sector data|eu institutions data protection"
     r"|data protection \(police\)|personal data protection"
     r"|privacy shield|standard contractual clause", 0),

    # ── 4. Cybersecurity & Operational Resilience ────────────────────
    (r"nis directive|nis2|cyber resilience|cyber solidarity|cyber sanctions"
     r"|attacks against information systems|eu.?lisa"
     r"|dora ict|network and information security"
     r"|radio equipment directive"
     # EU institution IT security regs
     r"|computer crime.*network|information security.*eu.*institution"
     r"|eu.*institution.*cybersecurity|eu.*institution.*information security", 4),

    # ── 1. AI, Data Economy & Digital Identity ───────────────────────
    (r"artificial intelligence|ai act|data act|data governance"
     r"|digital governance act|digital europe programme|interoperable europe"
     r"|chips act|eidas|electronic identif|digital identity"
     r"|digital single market|free flow of non-personal data"
     r"|eurohpc|high performance computing"
     r"|electronic signatures directive|database directive"
     r"|web accessibility directive|single digital gateway"
     r"|digitalisation of judicial|digitalisation cross.?border", 1),

    # ── 2. Platform Regulation & Digital Markets ─────────────────────
    (r"digital services act|digital markets act|dsa |dma |"
     r"electronic commerce|e.commerce directive|net neutrality"
     r"|roaming regulation|open data directive|gigabit infrastructure"
     r"|eecc|berec|p2b regulation|geo.?blocking"
     r"|cross.?border parcel delivery|postal services directive"
     r"|universal service directive|access directive 2002"
     r"|authorisation directive 2002|framework directive.*2002.*telecom"
     r"|regulation of telecommunications"        # old Framework Directive EuroVoc
     r"|infosoc directive|technical standards notification"
     r"|digital services act procedural"
     r"|online broadcasting directive"
     r"|broadband cost reduction"                # 2014/61
     r"|satellite communications.*equipment|r.tte directive"  # old radio/telecom terminal equipment
     r"|mobile.*communications.*market|mobile phone.*price.*retail"
     r"|retail.*roaming|roaming.*retail"
     r"|political advertising.*platform|advertising.*political.*online"
     r"|mass media.*political|political propaganda.*online", 2),

    # ── 3. Digital Content, Media & Copyright ────────────────────────
    (r"audiovisual media|avmsd|television without frontiers"
     r"|copyright|dsm copyright|orphan works|collective rights management"
     r"|marrakesh|rental and lending rights|copyright term"
     r"|database directive|online broadcasting"
     r"|media freedom act|emfa|european media freedom"
     r"|broadcasting|public service broadcasting", 3),

    # ── 8. Intellectual Property ─────────────────────────────────────
    (r"trade mark|trademark|community trade mark|eu trade mark"
     r"|designs directive|community designs|design regulation"
     r"|geographical indication|quality schemes regulation|craft gi"
     r"|trade secrets directive|intellectual property enforcement"
     r"|ipred|ohim|euipo|compulsory licens"
     r"|kimberley process"
     r"|spirit drinks regulation|quality\s+scheme"
     r"|industrial counterfeiting.*customs|customs.*ip enforcement"
     r"|customs inspection.*intellectual property"
     r"|prevention of infringements.*intellectual property"
     r"|intellectual property law", 8),

    # ── 9. Criminal Justice & Fundamental Rights ─────────────────────
    (r"criminal|terrorism|terrorist|europol|eurojust|eppo"
     r"|law enforcement|police cooperation|joint investigation team"
     r"|prüm|prum|asset recovery|confiscation|asset freeze"
     r"|anti.?trafficking|human trafficking|child sexual abuse"
     r"|firearms directive|firearms regulation"
     r"|firearms and munitions|arms control.*directive|illicit trade.*arms"
     r"|personal weapon.*exchange|personal weapon.*information"
     r"|olaf regulation|whistleblow|anti.?slapp"
     r"|e.?evidence|ecris|criminal records"
     r"|presumption of innocence|access to a lawyer|right to interpretation"
     r"|procedural safeguards for children|victims.? rights"
     r"|legal aid directive|anti.?torture trade"
     r"|law enforcement information exchange|pnr directive"
     r"|combating terrorism|attacks against information"
     r"|mutual legal assistance|european investigation order"
     r"|fundamental rights agency|rule of law conditionality"
     r"|compensation directive.*victim|global human rights sanctions"
     r"|freezing and confiscation orders"
     r"|eu global human rights|environmental crime directive"
     r"|fado regulation|false.*authentic document", 9),

    # ── 10. Migration, Asylum & Border Management ────────────────────
    (r"asylum|refugee|migration|schengen|border|visa "
     r"|eurodac|dublin regulation|frontex|european border and coast guard"
     r"|return directive|temporary protection directive"
     r"|reception conditions|qualification directive"
     r"|family reunification|long.?term residents|employer sanctions directive"
     r"|blue card directive|single permit directive|seasonal workers directive"
     r"|students and researchers directive|uniform visa format"
     r"|visa list regulation|eurosur|api regulation|api.pnr"
     r"|entry.?exit system|ees regulation|etias"
     r"|sis regulation|vis regulation|interoperability.*border"
     r"|rapid border intervention|migration statistics"
     r"|resettlement|asylum migration and integration fund"
     r"|border management fund|internal security fund"
     r"|return border procedure|crisis and force majeure regulation"
     r"|qualification regulation.*asylum|screening regulation"
     r"|consular protection directive"
     r"|passport.*security|biometric.*document|biometric.*passport"
     r"|forgery of documents.*border|secure identity card"
     r"|free movement.*persons.*identity|eu identity card"
     r"|residence permit.*biometric", 10),

    # ── 5. Financial Markets, Investment & Capital ───────────────────
    (r"mifid|mifir|emir|financial instrument|securities"
     r"|capital market|market abuse|prospectus directive|benchmark regulation"
     r"|esma|eba |eiopa|ccp recovery|short selling"
     r"|central securities|csdr|securitisation|sftr|mmf regulation"
     r"|aifmd|ucits|euveca|eusef|eltif|priips"
     r"|credit rating agenc|market in financial"
     r"|investment firm|investment fund directive"
     r"|remit |green bond|esg rating|listing act"
     r"|transparency directive.*securities|shareholders?\s+rights"
     r"|statutory audit|company law directive|digital company law"
     r"|nfrd|non.financial reporting|corporate sustainability reporting"
     r"|accounting directive|cross.border mergers directive"
     r"|takeover bids|financial conglomerates"
     r"|settlement finality|covered bonds regulation"
     r"|sustainable finance disclosure|eu climate benchmark|taxonomy regulation"
     r"|crowdfunding service|european crowdfunding|pepp regulation"
     r"|ifr regulation|european systemic risk board|esrb"
     r"|macroeconomic imbalances|ana.?credit"
     r"|dlT pilot|distributed ledger technology pilot"
     r"|mica |markets in crypto|esap|european single access point"
     r"|european sustainability reporting standards"
     r"|global minimum tax|country.by.country reporting"
     r"|tax dispute resolution|anti.tax avoidance directive"
     r"|dac6|dac7|administrative cooperation.*tax"
     r"|capital requirements regulation|crr "
     r"|merger regulation|antitrust enforcement|ecn plus"
     r"|damages actions directive|general block exemption"
     r"|foreign subsidies regulation.*procedural"
     r"|efsi regulation|european fund.*strategic investment"
     # old company law directives (caught by company law EuroVoc/directory)
     r"|public limited company|private limited company|company.*demerger"
     r"|publication of accounts.*company|branch.*disclosure.*company"
     r"|corporate governance.*directive|proxy vote.*directive"
     # ECB/monetary implementing regs
     r"|european central bank.*regulation|ecb.*sanction|ecb.*supervision"
     r"|european system of central banks.*collection|escb.*statistics"
     # EMU instruments
     r"|coordination of emu|instruments of economic policy.*euro"
     r"|euro group.*stabilisation|eurozone.*governance"
     # crypto/virtual currency (MiCA Level 2)
     r"|virtual currency.*technical|issuing of currency.*standard"
     r"|free movement of capital.*crypto|capital.*virtual currency"
     # Level 2 financial technical standards (generic)
     r"|benchmarking.*financial|financial.*benchmarking|euribor|libor"
     r"|financial legislation.*technical standard|technical standard.*financial market"
     r"|financial supervision.*technical|financial.*technical.*standard", 5),

    # ── 6. Banking, Payments, Insurance & AML ────────────────────────
    (r"crd iv|crd v|crd vi|capital requirements directive"
     r"|brrd|bank recovery|resolution directive|single resolution"
     r"|deposit guarantee|dgsd"
     r"|psd1|psd2|payment services directive|payment accounts"
     r"|instant payments|sepa regulation|interchange fees"
     r"|anti.money laundering|aml |fourth anti.money laundering"
     r"|fifth anti.money laundering|sixth anti.money laundering"
     r"|transfer of funds regulation|cash controls regulation"
     r"|consumer credit directive|mortgage credit directive"
     r"|distance marketing.*financial|financial collateral"
     r"|insurance distribution|solvency ii|iorp directive|iorp ii"
     r"|insurance recovery|insurance mediation"
     r"|e.money directive|electronic money"
     r"|credit servicers|bank restructuring|credit institution"
     r"|vat directive|vat administrative|vat rules|turnover tax|vat rate"
     r"|motor insurance directive|winding up directive"
     r"|financial conglomerates|ssm regulation|ssm framework"
     r"|faster withholding tax"
     r"|covered bonds directive|covered bonds regulation"
     r"|preventing.*money laundering|money laundering.*criminal"
     r"|non.cash payment fraud"
     r"|preventive restructuring directive|insolvency protection"
     r"|omnibus ii directive"
     # old insurance (life assurance, insurance company directives)
     r"|life assurance.*directive|insurance company.*directive"
     r"|directive.*insurance company|directive.*life assurance"
     # bank resolution old
     r"|winding up.*bank|bank.*winding up|winding up.*credit institution"
     # SSM implementing
     r"|ecb.*banking supervision|banking supervision.*ecb|eu banking union"
     # capital adequacy old
     r"|capital adequacy directive"
     # VAT implementing
     r"|tax.*information exchange|prevention of tax evasion"
     r"|tax cooperation|direct tax.*directive"
     # payment data (SEPA-related implementing)
     r"|payment system.*data|data.*payment system", 6),

    # ── 7. Consumer Rights & Internal Market ─────────────────────────
    (r"consumer rights directive|omnibus directive|class action"
     r"|consumer odr|alternative dispute resolution.*consumer"
     r"|unfair commercial practices|unfair terms directive"
     r"|consumer sales|distance selling directive"
     r"|package travel|timeshare|price indication"
     r"|product liability|general product safety directive"
     r"|toy safety|toys safety|food information to consumers"
     r"|services directive.*internal market|professional qualifications"
     r"|internal market information|imi regulation"
     r"|procurement|concessions directive|public procurement"
     r"|defence procurement|common procurement vocabulary"
     r"|electronic invoicing.*procurement|e.invoicing"
     r"|unfair trading practices"
     r"|single market|internal market emergency"
     r"|accreditation.*market surveillance|market surveillance regulation"
     r"|construction products regulation|machinery regulation"
     r"|lifts directive|pressure equipment directive"
     r"|atex directive|low voltage directive|measuring instruments"
     r"|emc directive|electromagnetic compatibility"
     r"|pyrotechnic articles|recreational craft"
     r"|biocidal products regulation"
     r"|cosmetic products regulation"
     r"|textile labelling|tyre labelling"
     r"|food contact materials|food additives regulation"
     r"|novel food|irradiated food|flavourings"
     r"|rome i regulation|rome ii regulation|rome iii regulation"
     r"|brussels i |brussels ii |succession regulation"
     r"|service of documents|evidence taking regulation|taking of evidence"
     r"|small claims|european order for payment|european enforcement order"
     r"|european account preservation|insolvency regulation"
     r"|legal aid.*civil|mediation directive|injunctions directive"
     r"|right to repair|empowering consumers"
     r"|cross.border mobility directive.*company"
     r"|digital company law|company law directive"
     r"|pay transparency|women on boards|equal pay"
     r"|work.life balance|working time directive|fixed.term work"
     r"|temporary agency work|information and consultation directive"
     r"|european works councils|acquired rights directive"
     r"|seasonal workers.*internal|free movement.*workers"
     r"|workers free movement|equal treatment.*employment"
     r"|employment equality|racial equality directive"
     r"|gender.*goods and services|self.employed equal treatment"
     r"|posting.*workers|labour authority"
     r"|platform work|pay transparency|adequate minimum wages"
     r"|european labour authority|eures"
     r"|mutual recognition of goods|european standardization"
     r"|standardisation regulation|technical standards notification"
     r"|geo.?blocking|cross.border parcel"
     r"|foreign subsidies regulation|fdi screening"
     r"|antitrust.*enforcement|ecn.?plus directive|damages actions"
     r"|general block exemption|competition"
     r"|european citizens.*initiative|citizens initiative"
     r"|product safety directive|market access"
     r"|single.use plastics|plastic bags directive"
     r"|european disability card|european accessibility"
     r"|disability|accessibility act"
     r"|insolvency protection directive|winding up"
     # product safety technical directives (caught by EuroVoc)
     r"|ec conformity marking|ce marking|conformity assessment.*product"
     r"|personal protective equipment|ppe regulation"
     r"|transparent.*working conditions|predictable working conditions"
     r"|lawyers establishment|legal profession.*directive"
     r"|recognition of diplomas.*profession"
     # social security coordination
     r"|social security.*coordination|coordination.*social security"
     r"|social.security.*migrant|migrant.*social security"
     r"|social.security harmonisation"
     # dangerous substances (workplace/product safety angle)
     r"|dangerous substance.*classification|chemical.*classification.*label"
     r"|dangerous preparations directive|chemical.*product.*safety"
     # old freedom of establishment directives (companies)
     r"|freedom of establishment.*directive.*company"
     r"|freedom of establishment.*lawyer", 7),

    # ── 11. Transport & Mobility ──────────────────────────────────────
    (r"transport|aviation|maritime|shipping|rail |railway"
     r"|driving licences|driving hours|roadworthiness|roadside inspection"
     r"|tachograph|motor vehicle|vehicle registration"
     r"|road safety|traffic offences|ecall|eets"
     r"|airport slots|air services|air passenger|flight compensation"
     r"|air carrier blacklist|air carrier black list"
     r"|easa|emsa regulation|marine.*safety"
     r"|ship.*pollution|vessel traffic monitoring|port state control"
     r"|ship.*security|port.*facility security|ship reporting"
     r"|maritime labour|seafarers|inland waterways|inland navigation"
     r"|passenger.*sea|maritime passenger|ship recycling|fueleu maritime"
     r"|maritime single window|maritime spatial planning|marine strategy"
     r"|marine equipment directive"
     r"|professional drivers directive|train drivers"
     r"|civil aviation accident|aviation security regulation"
     r"|ten.t|trans.european.*transport"
     r"|alternative fuels infrastructure|euro 5|euro 6|euro 7|euro vi"
     r"|heavy.duty vehicles co2|motor vehicle type approval"
     r"|general vehicle safety|uas.*regulation|uas operations"
     r"|unmanned aircraft|drones regulation"
     r"|mmtis|multimodal travel"
     r"|its directive|intelligent transport"
     r"|river information|ris directive"
     r"|transportable pressure equipment"
     r"|posting of drivers|cross.border.*traffic"
     r"|nrmm regulation|non.road mobile machinery", 11),

    # ── 12. Environment, Energy & Climate ────────────────────────────
    (r"environment|climate|energy|emission|carbon|greenhouse gas"
     r"|waste |landfill|incineration|packaging.*waste|waste framework"
     r"|waste shipment|waste incineration|mining waste|ship recycling"
     r"|water directive|drinking water|bathing water|water reuse"
     r"|wastewater|urban waste water|nitrates directive"
     r"|marine strategy framework|marine.*pollution"
     r"|nature restoration|habitats directive|birds directive|wildlife trade"
     r"|biodiversity|ecology|persistent organic pollutants"
     r"|deforestation|timber regulation|lulucf"
     r"|ozone regulation|ozone layer"
     r"|f.gas|fluorinated|pcb directive"
     r"|reach regulation|chemicals regulation|clp regulation"
     r"|biocidal products directive"
     r"|seveso|industrial emissions|ippc directive"
     r"|ecodesign directive|energy labelling directive|energy performance"
     r"|energy efficiency directive|renewable energy directive"
     r"|gas directive|gas transmission|gas storage|gas security"
     r"|gas solidarity|repowereu"
     r"|electricity.*directive|electricity.*regulation|electricity market"
     r"|oil stocks directive|hydrocarbons"
     r"|cbam regulation|carbon border"
     r"|carbon removal|carbon offset"
     r"|ten.e|trans.european.*energy|energy union"
     r"|just transition|social climate fund|european climate law"
     r"|ecolabel|emas regulation|pollutant release"
     r"|ambient air quality|air quality directive"
     r"|environmental information access"
     r"|sustainable use of pesticides"
     r"|floods directive|flood risk"
     r"|ccs directive|carbon capture|carbon storage"
     r"|fuel quality|petrol vapour|solvents emissions"
     r"|batteries regulation|battery regulation"
     r"|soil monitoring|soil law"
     r"|critical raw materials"
     r"|plant protection products regulation"
     r"|organic production regulation"
     r"|life programme|european green deal", 12),

    # ── 13. Health, Medicines & Life Sciences ────────────────────────
    (r"health|medicine|medicinal product|pharmaceutical"
     r"|clinical trial|medical device|in vitro diagnostic"
     r"|ema regulation|ecdc regulation|hera"
     r"|pharmacovigilance|falsified medicines|cross.border health"
     r"|organ donation|blood safety|human tissue|substances of human origin"
     r"|tobacco products|tobacco advertising|smoking"
     r"|orphan medicines|paediatric regulation"
     r"|veterinary medicinal|veterinary.*medicine"
     r"|food hygiene|food safety|general food law|salmonella"
     r"|tse regulation|bse|prion"
     r"|quick.frozen foods|irradiated food|novel food|flavourings regulation"
     r"|food additives|food information to consumers|food contact materials"
     r"|gmo.*traceability|gmo.*labelling"
     r"|feed additives|animal feed"
     r"|occupational health|noise at work|vibration directive"
     r"|artificial optical radiation|electromagnetic fields directive"
     r"|medical countermeasures|emergency.*framework.*health"
     r"|eu4health|covid.*certificate|digital covid"
     r"|euda|drug monitoring centre|emcdda"
     r"|public health|patient rights|cross.border healthcare"
     r"|medicinal products for human|medicines directive"
     r"|good laboratory practice"
     # old health/foodstuff directives (EuroVoc: foodstuff, sweetener, solvent)
     r"|foodstuff.*directive|food.*solvent|food.*sweetener|sweetener.*food"
     r"|foodstuffs.*legislation|food.*standard.*directive"
     # nursing/medical training
     r"|nursing staff.*training|recognition of diplomas.*nurs"
     r"|medical training.*recognition", 13),

    # ── 14. Agriculture, Fisheries & Rural Development ───────────────
    (r"fisheries|fishing|aquaculture|fish stock|fish catch"
     r"|cfp |common fisheries policy|gfcm|nafo|ccamlr"
     r"|mediterranean fisheries|atlantic.*fish|iuu fishing"
     r"|emff|emfaf|european fisheries fund|fisheries control"
     r"|agriculture|agricultural|rural development"
     r"|cap |common agricultural policy|cmo regulation|single cmo"
     r"|arable|crop|cereals|sugar|wine |spirits regulation"
     r"|animal health law|animal transport|animal breed"
     r"|livestock|cattle|swine|poultry|laying hens|calves directive"
     r"|veterinary.*legislation|plant health|plant protection"
     r"|official controls regulation|food.*hygiene"
     r"|organic production|organic farming"
     r"|animal by.products|slaughter regulation"
     r"|gmo.*traceability|gmo.*food"
     r"|rural|farm data|farm sustainability|fsdn"
     r"|food aid|fead regulation"
     r"|agri.statistics|agricultural statistics|posei"
     r"|quality schemes|geographical indication.*food"
     r"|spirit drinks|beer|wine sector"
     r"|deforestation.*supply chain"
     r"|sustainable food system"
     r"|maximum residue levels"
     # sericulture / silkworms (niche agri)
     r"|sericulture|silkworm", 14),

    # ── 15. EU Institutional Framework & Funding ─────────────────────
    (r"horizon europe|research programme|research framework"
     r"|erasmus|creative europe|european social fund|esf\+"
     r"|common provisions regulation.*structural|erdf|cohesion fund"
     r"|interreg regulation|connecting europe facility|cef regulation"
     r"|rrf |recovery.*resilience facility|react.eu|multiannual financial framework"
     r"|investeu|efsi regulation|brexit adjustment"
     r"|global europe|ipa iii|instrument.*stability"
     r"|development cooperation instrument|humanitarian aid regulation"
     r"|ukraine facility|step regulation"
     r"|eu financial regulation|financial regulation.*eu"
     r"|staff regulations|comitology|european ombudsman"
     r"|eige|european institute.*gender"
     r"|fundamental rights agency"
     r"|european institute of innovation|eit regulation"
     r"|etf regulation|european training foundation"
     r"|galileo regulation|space programme|copernicus"
     r"|union secure connectivity|eutelsat"
     r"|dual.use.*export|export control"
     r"|anti.coercion instrument|foreign subsidies.*instrument"
     r"|forced labour regulation"
     r"|european statistics|nace regulation|eurostat"
     r"|statistics.*income|statistics.*living|statistics.*business"
     r"|inspire directive|spatial data infrastructure"
     r"|geographical information system|inspire.*implement"
     r"|edirpa|edip regulation|european defence"
     r"|security action for europe|european peace facility"
     r"|customs.*programme|customs code|modernised customs|union customs"
     r"|cerv programme|citizens.*values"
     r"|egf regulation|european globalisation adjustment"
     r"|rule of law conditionality"
     r"|european citizens initiative"
     r"|european political parties"
     r"|public country.by.country|country.by.country reporting"
     r"|dac |administrative cooperation.*tax"
     r"|european parliament elections"
     r"|state aid|general block exemption"
     r"|consular protection"
     # CFSP / foreign policy / sanctions
     r"|common foreign and security policy|european political cooperation"
     r"|international sanctions|eu restrictive measure|restrictive measures"
     r"|macro.financial assistance|european neighbourhood"
     r"|pre.accession|candidate.*country.*aid|ipa.*regulation"
     # EU institutional governance
     r"|access to eu information|access to documents.*eu"
     r"|eu official journal|electronic publishing.*eu|official journal.*eu"
     r"|powers of the institutions|capacity to exercise rights"
     r"|operation of the institutions"
     r"|eu budget.*control|budgetary control|own resources.*eu"
     r"|eu publication.*electronic|publishing.*eu"
     # statistics (generic EU statistics regs)
     r"|eu statistics|statistical nomenclature|statistical method.*eu"
     r"|purchasing power parity.*eu|population census.*eu"
     r"|trade statistics.*eu|external trade statistics"
     r"|european business statistics"
     # EU agencies / programmes
     r"|eurofound|cedefop|european training"
     r"|court of justice.*restructur|court of justice.*reform"
     r"|european citizenship|citizenship of the union"
     r"|european election|european parliament election"
     r"|social protection committee|structural reform support"
     r"|european solidarity corps|volunteering.*eu"
     r"|european.*sport|sport.*eu programme"
     # anti-dumping / trade defense (Common Commercial Policy)
     r"|anti.?dumping duty|anti.?dumping measure|countervailing charge|countervailing dut"
     r"|anti.?subsidy|surveillance concerning imports|safeguard.*import"
     r"|originating product.*import.*anti|import.*originating.*anti"
     # cultural objects (EU external culture/heritage policy)
     r"|cultural object.*import|heritage protection.*illicit"
     r"|illicit trade.*cultural.*object|import.*cultural.*heritage"
     # tax harmonization (old)
     r"|tax evasion.*directive|tax harmonisation.*directive|direct tax.*cooperation"
     r"|elimination of double taxation|double taxation.*convention"
     # capital movement old
     r"|free movement of capital.*directive"
     # EU budget / financial management
     r"|general budget.*eu|eu.*general budget|financial management.*eu.*budget"
     r"|fund.*eu.*credit guarantee|community loan"
     r"|eu.*programme.*research|research.*eu.*programme"
     r"|eu.*programme.*youth|youth.*eu.*programme"
     # company law (old) → institutional/market
     r"|company law$|company law[^d\-]"
     # Demographic / social statistics
     r"|demographic statistics|population.*statistics.*eu"
     r"|migration statistics.*eu", 15),
]

# ─────────────────────────────────────────────────────────────────────────────
# Build the compiled rule list once
# ─────────────────────────────────────────────────────────────────────────────
COMPILED_RULES = [(re.compile(pat, re.IGNORECASE), cid) for pat, cid in KEYWORD_RULES]


def build_haystack(readable: str, node: dict) -> str:
    """Concatenate all available text fields for a law node into one string."""
    parts = [readable]
    parts.append(node.get("name", ""))
    sm = node.get("subject_matter") or []
    if isinstance(sm, list):
        parts.extend(sm)
    elif isinstance(sm, str):
        parts.append(sm)
    ev = node.get("eurovoc") or []
    if isinstance(ev, list):
        parts.extend(ev)
    elif isinstance(ev, str):
        parts.append(ev)
    dr = node.get("directory") or []
    if isinstance(dr, list):
        parts.extend(dr)
    elif isinstance(dr, str):
        parts.append(dr)
    return " | ".join(str(p) for p in parts if p)


def keyword_classify(haystack: str) -> int:
    """Return the first matching cluster id, or 19 (Other) if nothing matches."""
    for pattern, cid in COMPILED_RULES:
        if pattern.search(haystack):
            return cid
    return 19


def main():
    # Load readable names
    names: dict[str, str] = {}
    with open(NAMES_CSV, encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)   # header
        for row in reader:
            if len(row) >= 2:
                names[row[0].strip()] = row[1].strip()

    # Load layout
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
    law_nodes: list[dict] = layout["law_nodes"]
    node_index = {n["id"]: n for n in law_nodes}
    print(f"Loaded {len(law_nodes)} law nodes")

    # Classify
    cluster_counts: dict[int, int] = {}
    method_counts = {"hand": 0, "keyword": 0, "other": 0}
    rows = []

    for n in sorted(law_nodes, key=lambda x: x["id"]):
        celex   = n["id"]
        readable = names.get(celex, n.get("name", "")[:120])

        if celex in CLUSTER_MAP:
            cid    = CLUSTER_MAP[celex]
            method = "hand"
            method_counts["hand"] += 1
        else:
            haystack = build_haystack(readable, n)
            cid      = keyword_classify(haystack)
            if cid == 19:
                method = "other"
                method_counts["other"] += 1
            else:
                method = "keyword"
                method_counts["keyword"] += 1

        cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
        cname = CLUSTER_NAMES[cid]
        rows.append([celex, readable, cid, cname, method])

    # Write CSV
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["celex", "readable_name", "cluster_id", "cluster_name", "method"])
        w.writerows(rows)

    print(f"\nWrote -> {OUT}")
    print(f"\nClassification method breakdown:")
    print(f"  Hand-mapped:      {method_counts['hand']:5}")
    print(f"  Keyword-matched:  {method_counts['keyword']:5}")
    print(f"  Still 'Other':    {method_counts['other']:5}")
    print(f"\nCluster sizes:")
    for cid in sorted(cluster_counts):
        marker = " <-- TARGET" if cid in (0, 1, 2, 4) else ""
        print(f"  [{cid:2}] {CLUSTER_NAMES[cid]:48} {cluster_counts[cid]:4} laws{marker}")

    # Show remaining unclassified
    still_other = [(r[0], r[1]) for r in rows if r[2] == 19]
    print(f"\nStill unclassified ({len(still_other)} laws):")
    for celex, name in still_other[:60]:
        print(f"  {celex} | {name[:100]}")
    if len(still_other) > 60:
        print(f"  ... and {len(still_other) - 60} more")


if __name__ == "__main__":
    main()
