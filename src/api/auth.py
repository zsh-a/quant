"""
JWT 认证模块。

启用方式: 设置环境变量 QUANT_AUTH__ENABLED=true
配置项:
  QUANT_AUTH__SECRET_KEY        — JWT 签名密钥（生产环境必须修改）
  QUANT_AUTH__DEFAULT_USERNAME  — 默认用户名
  QUANT_AUTH__DEFAULT_PASSWORD  — 默认密码
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel

from src.config.settings import get_auth_config

router = APIRouter(prefix="/auth", tags=["auth"])

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer = HTTPBearer(auto_error=False)


# ------------------------------------------------------------------
# Token helpers
# ------------------------------------------------------------------

def _create_access_token(sub: str, expires_delta: Optional[timedelta] = None) -> str:
    cfg = get_auth_config()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=cfg.access_token_expire_minutes)
    )
    payload = {"sub": sub, "exp": expire}
    return jwt.encode(payload, cfg.secret_key, algorithm=cfg.algorithm)


def _verify_token(token: str) -> str:
    """Return the *sub* claim or raise 401."""
    cfg = get_auth_config()
    try:
        payload = jwt.decode(token, cfg.secret_key, algorithms=[cfg.algorithm])
        sub: str | None = payload.get("sub")
        if sub is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "无效的令牌")
        return sub
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "令牌已过期")
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "无效的令牌")


# ------------------------------------------------------------------
# FastAPI dependency — noop when auth is disabled
# ------------------------------------------------------------------

async def require_auth(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[str]:
    """FastAPI dependency: returns username or *None* when auth is disabled."""
    cfg = get_auth_config()
    if not cfg.enabled:
        return None
    if creds is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "需要认证",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _verify_token(creds.credentials)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    """用户名密码换取 JWT token。"""
    cfg = get_auth_config()
    if not cfg.enabled:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "认证未启用")

    # 简单的单用户模式；可扩展为数据库查询
    if body.username != cfg.default_username or body.password != cfg.default_password:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "用户名或密码错误")

    token = _create_access_token(body.username)
    return TokenResponse(
        access_token=token,
        expires_in=cfg.access_token_expire_minutes * 60,
    )


@router.get("/me")
async def me(username: Optional[str] = Depends(require_auth)):
    """返回当前用户信息（也可用于验证 token 是否有效）。"""
    cfg = get_auth_config()
    return {
        "auth_enabled": cfg.enabled,
        "username": username,
    }
