import re


_INVALID_DOTTED_ID_RE = re.compile(r'[\s/\\]')


def validate_dotted_identifier(value: str, *, kind: str = 'identifier') -> str:
    """
    Validate a dotted identifier used by DAGs, node groups, and node ids.

    Dots are allowed and represent logical hierarchy, but each component must be
    a non-empty plain identifier segment. Whitespace and path separators are
    rejected so identifiers can be safely mapped to artifact directories.
    """
    if not isinstance(value, str) or value == '':
        raise ValueError(f'{kind} must be a non-empty string')

    if _INVALID_DOTTED_ID_RE.search(value):
        raise ValueError(
            f'{kind} cannot contain whitespace or path separators: {value!r}'
        )

    parts = value.split('.')
    if any(part == '' for part in parts):
        raise ValueError(
            f'{kind} cannot contain empty dotted components: {value!r}'
        )

    return value
