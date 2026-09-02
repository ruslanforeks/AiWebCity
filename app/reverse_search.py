from __future__ import annotations
import asyncio
import re
from typing import Any
from PicImageSearch import GoogleLens, Network, Yandex
REVERSE_SEARCH_TIMEOUT=45.0
REVERSE_SEARCH_RESULTS=20
def norm_text(value: Any)->str:
    return re.sub(r"\s+"," ",str(value or "")).strip()
def extract_year(text:str)->int|None:
    years=[int(x) for x in re.findall(r"(?<!\d)(1[5-9]\d{2}|20\d{2})(?!\d)",text)]
    return min(years) if years else None
def item_to_dict(item:Any,source:str)->dict[str,Any]:
    title=norm_text(getattr(item,"title","")); page_url=norm_text(getattr(item,"url","")); thumb=norm_text(getattr(item,"thumbnail","")); desc=norm_text(getattr(item,"content","")); site=norm_text(getattr(item,"source","")); size=norm_text(getattr(item,"size",""))
    return {"image_url":thumb,"page_url":page_url,"title":title or f"Результат {source}","description":desc,"source":source,"kind":"reverse_image","site":site,"size":size,"year":extract_year(" ".join([title,desc,page_url]))}
def dedupe(items:list[dict[str,Any]],limit:int=40)->list[dict[str,Any]]:
    seen=set(); out=[]
    for item in items:
        key=norm_text(item.get("page_url")) or norm_text(item.get("image_url"))
        if not key or key in seen: continue
        seen.add(key); out.append(item)
        if len(out)>=limit: break
    return out
async def yandex_search(image_bytes:bytes)->dict[str,Any]:
    try:
        async with Network(timeout=REVERSE_SEARCH_TIMEOUT) as client:
            engine=Yandex(client=client,base_url="https://yandex.com")
            response=await asyncio.wait_for(engine.search(file=image_bytes),timeout=REVERSE_SEARCH_TIMEOUT)
        return {"engine":"Yandex Images","ok":True,"results":[item_to_dict(i,"Yandex Images") for i in (response.raw or [])][:REVERSE_SEARCH_RESULTS],"search_url":norm_text(getattr(response,"url","")) or None,"error":None}
    except Exception as exc:
        return {"engine":"Yandex Images","ok":False,"results":[],"search_url":None,"error":f"{type(exc).__name__}: {str(exc)[:300]}"}
async def google_lens_search(image_bytes:bytes)->dict[str,Any]:
    try:
        async with Network(timeout=REVERSE_SEARCH_TIMEOUT) as client:
            engine=GoogleLens(client=client,search_type="all",hl="ru",country="RU")
            response=await asyncio.wait_for(engine.search(file=image_bytes),timeout=REVERSE_SEARCH_TIMEOUT)
        return {"engine":"Google Lens","ok":True,"results":[item_to_dict(i,"Google Lens") for i in (response.raw or [])][:REVERSE_SEARCH_RESULTS],"search_url":norm_text(getattr(response,"url","")) or None,"error":None}
    except Exception as exc:
        return {"engine":"Google Lens","ok":False,"results":[],"search_url":None,"error":f"{type(exc).__name__}: {str(exc)[:300]}"}
async def search(image_bytes:bytes)->dict[str,Any]:
    yandex,lens=await asyncio.gather(yandex_search(image_bytes),google_lens_search(image_bytes))
    return {"enabled":True,"engines":[yandex,lens],"results":dedupe(yandex["results"]+lens["results"])}
