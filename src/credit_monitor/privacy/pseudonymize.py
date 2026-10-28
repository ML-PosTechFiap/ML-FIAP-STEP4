"""Pseudonimizacao de SK_ID_CURR por HMAC-SHA256 com salt secreto."""

from __future__ import annotations

import hashlib
import hmac

import pandas as pd

ID_COLUMN = "SK_ID_CURR"
KEY_COLUMN = "client_key"
KEY_LENGTH = 16
# Valor publico de `.env.example`: quem o usa deixa as chaves reversiveis por forca bruta
# (SK_ID_CURR e um inteiro pequeno), o que equivale a nao ter salt.
PLACEHOLDER_SALT = "troque-este-valor"


class InvalidSalt(ValueError):
    """`PSEUDONYMIZATION_SALT` vazio ou igual ao valor público de exemplo."""


def _require_salt(salt: str) -> None:
    if not salt:
        raise InvalidSalt("PSEUDONYMIZATION_SALT vazio: defina um salt secreto no .env")
    if salt == PLACEHOLDER_SALT:
        raise InvalidSalt(
            f"PSEUDONYMIZATION_SALT ainda é o valor de exemplo ({PLACEHOLDER_SALT!r}), "
            "público no repositório: as chaves seriam reversíveis. Gere um salt próprio com "
            '`python -c "import secrets; print(secrets.token_hex(32))"` e grave no .env.'
        )


def _key(raw_id: object, salt: str) -> str | None:
    if pd.isna(raw_id):
        return None
    digest = hmac.new(salt.encode("utf-8"), str(int(raw_id)).encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()[:KEY_LENGTH]


def client_key(raw_id: object, salt: str) -> str | None:
    """Chave pseudonima de um ID; `None` quando o ID e nulo (None, NaN ou pd.NA)."""
    _require_salt(salt)
    return _key(raw_id, salt)


def pseudonymize(df: pd.DataFrame, salt: str) -> pd.DataFrame:
    """Copia de `df` com `client_key` como primeira coluna, no lugar de SK_ID_CURR."""
    _require_salt(salt)
    if ID_COLUMN not in df.columns:
        raise ValueError(f"coluna {ID_COLUMN} ausente: nada a pseudonimizar")
    ids = df[ID_COLUMN]
    mapping = {raw: _key(raw, salt) for raw in ids.dropna().unique()}
    keys = pd.Series(ids.map(mapping), index=df.index, dtype="object")
    keys = keys.where(keys.notna(), None)
    out = df.drop(columns=[ID_COLUMN])
    out.insert(0, KEY_COLUMN, keys)
    return out
