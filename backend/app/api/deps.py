"""路由公共依赖。"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.base import get_db

SessionDep = Annotated[Session, Depends(get_db)]
