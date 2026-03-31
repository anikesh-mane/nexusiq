class DocumentClassification(BaseModel):
    document_type: str  # invoice | contract | email | report | other
    confidence: float
    reasoning: str


class LineItem(BaseModel):
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None


class InvoiceExtraction(BaseModel):
    vendor: Optional[str] = None
    buyer: Optional[str] = None
    invoice_number: Optional[str] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    date: Optional[str] = None
    due_date: Optional[str] = None
    line_items: list[LineItem] = []
    payment_terms: Optional[str] = None
    tax: Optional[float] = None


class ContractExtraction(BaseModel):
    parties: list[str] = []
    effective_date: Optional[str] = None
    expiry_date: Optional[str] = None
    value: Optional[float] = None
    jurisdiction: Optional[str] = None
    key_obligations: list[str] = []
    termination_clause: Optional[str] = None


class EmailExtraction(BaseModel):
    sender: Optional[str] = None
    recipients: list[str] = []
    subject: Optional[str] = None
    date: Optional[str] = None
    action_items: list[str] = []
    sentiment: Optional[str] = None


class GenericExtraction(BaseModel):
    fields: dict[str, Any] = {}


# Map document types to their extraction schema
EXTRACTION_SCHEMAS: dict[str, type[BaseModel]] = {
    "invoice": InvoiceExtraction,
    "contract": ContractExtraction,
    "email": EmailExtraction,
    "report": GenericExtraction,
    "other": GenericExtraction,
}
