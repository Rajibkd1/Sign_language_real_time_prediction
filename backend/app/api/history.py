from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas import HistoryResponse


router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=HistoryResponse)
def list_history(request: Request) -> HistoryResponse:
    history = request.app.state.history_service
    return HistoryResponse(items=history.list())


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def clear_history(request: Request):
    request.app.state.history_service.clear()
    return None


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_history_item(item_id: str, request: Request):
    deleted = request.app.state.history_service.delete(item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="History item not found.")
    return None

