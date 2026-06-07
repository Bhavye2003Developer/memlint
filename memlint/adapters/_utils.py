from datetime import datetime
from dateutil import parser as dateutil_parser


def parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    return dateutil_parser.parse(value).replace(tzinfo=None)
