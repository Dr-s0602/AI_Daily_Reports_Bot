import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

from duckduckgo_search import DDGS
import arxiv
import google.generativeai as genai


# -----------------------------
# 0) 고정 포맷 템플릿(v1)
# -----------------------------
REPORT_TEMPLATE = """# [AI Daily] {target_date} 기술 동향

> 생성 시간(KST): {generated_time_kst}
> 데이터 소스: DuckDuckGo(뉴스 {news_n}), arXiv(cs.AI/cs.LG, 논문 {paper_n})

## 오늘의 Top 이슈 (3~5)
- 1) {{이슈명}} — {{핵심 키워드 3개}}
- 2) {{이슈명}} — {{핵심 키워드 3개}}
- 3) {{이슈명}} — {{핵심 키워드 3개}}
- (옵션) 4) ...
- (옵션) 5) ...

---

## 1. {{이슈명}}
### 요약
- ...
### 개발자 관점 한 줄 평
- ...
### 지금 바로 적용 아이디어
- ...
### 리스크/주의
- ...
### 참고 링크
- [..](..)
- [..](..)

---

## 2. {{이슈명}}
(동일 구조 반복)

---

## 오늘의 실무 액션 3가지
1) ...
2) ...
3) ...

## 원문 목록 (Raw Index)
### 뉴스
- {{title}} — {{url}}

### 논문
- {{title}} — {{pdf_url}}
"""


# -----------------------------
# 1) 공용 유틸
# -----------------------------
def generate_with_retry(model_name: str, prompt_text: str, retries: int = 3):
    """
    Gemini 호출 함수.
    - 429/quota 류 에러에 대해 30s, 60s 백오프로 재시도.
    - 그 외 에러는 즉시 raise.
    """
    model = genai.GenerativeModel(model_name)
    last_err = None

    for i in range(retries):
        try:
            return model.generate_content(prompt_text)
        except Exception as e:
            last_err = e
            msg = str(e).lower()

            if ("429" in msg or "quota" in msg) and i < retries - 1:
                wait_time = (i + 1) * 30
                print(f"⚠️ 레이트/쿼터 제한 추정: {wait_time}초 후 재시도 ({i+1}/{retries})")
                time.sleep(wait_time)
            else:
                raise

    raise last_err


def choose_model():
    """
    실행 시점에 사용 가능한 모델 중 우선순위로 선택.
    """
    all_models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
    candidates = [
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-1.5-flash",
        "models/gemini-pro",
    ]
    return next((c for c in candidates if c in all_models), all_models[0])


def build_raw_index(items):
    """
    Raw Index는 모델이 만들게 하지 말고 코드가 생성해서 프롬프트에 주입.
    링크 누락 방지 목적.
    """
    news_lines = []
    paper_lines = []

    for it in items:
        t = (it.get("title") or "").strip()
        link = (it.get("link") or "").strip()
        if not t or not link:
            continue

        if it.get("type") == "news":
            news_lines.append(f"- {t} — {link}")
        elif it.get("type") == "paper":
            paper_lines.append(f"- {t} — {link}")

    return "\n".join(news_lines), "\n".join(paper_lines)


def validate_report_format(report_text: str):
    """
    리포트가 템플릿 핵심 헤더를 누락했는지 검사.
    """
    required = [
        "# [AI Daily]",
        "## 오늘의 Top 이슈",
        "## 오늘의 실무 액션 3가지",
        "## 원문 목록 (Raw Index)",
        "### 뉴스",
        "### 논문",
    ]
    return [k for k in required if k not in report_text]


def looks_truncated(report_text: str):
    """
    무료 환경에서 출력이 중간에 끊겼는지 간단 휴리스틱으로 감지.
    """
    if len(report_text) < 1200:
        return True
    if "## 원문 목록 (Raw Index)" not in report_text:
        return True
    return False


def continue_report(model_name: str, existing_text: str):
    """
    리포트가 끊긴 경우 1회 이어쓰기.
    """
    tail = existing_text[-1500:]
    prompt2 = f"""
아래 글의 다음 내용을 이어서 작성하라. 중복/재작성 금지.
마크다운 형식 유지. 마지막은 자연스럽게 마무리.

--- 글의 끝부분 ---
{tail}
--- 여기부터 이어쓰기 ---
""".strip()
    resp2 = generate_with_retry(model_name, prompt2, retries=3)
    return existing_text + "\n" + resp2.text.strip()


# -----------------------------
# 2) 수집 (뉴스 5 + 논문 5)
# -----------------------------
def collect_items(target_date: str, news_n: int = 5, paper_n: int = 5):
    items = []

    # DuckDuckGo
    with DDGS() as ddgs:
        results = ddgs.text(f"AI technology news {target_date}", max_results=news_n)
        for r in results:
            items.append(
                {
                    "type": "news",
                    "title": (r.get("title") or "").strip(),
                    "body": (r.get("body") or "").strip(),
                    "link": (r.get("href") or "").strip(),
                }
            )

    # arXiv
    search = arxiv.Search(
        query="cat:cs.AI OR cat:cs.LG",
        max_results=paper_n,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    for result in search.results():
        items.append(
            {
                "type": "paper",
                "title": (result.title or "").strip(),
                "body": (result.summary or "").strip(),
                "link": (result.pdf_url or "").strip(),
            }
        )

    return items


# -----------------------------
# 3) Map (1차 구조화 요약)
# -----------------------------
def map_summaries(model_name: str, items):
    summaries = []

    for idx, it in enumerate(items, start=1):
        item_text = f"""
[타입] {it['type']}
[제목] {it['title']}
[본문]
{it['body']}
[링크] {it['link']}
""".strip()

        summary_prompt = f"""
너는 시니어 개발자 관점의 AI 뉴스/논문 분석가다.
아래 항목을 한국어로 '구조화 요약'하라.
사실 중심으로 작성하고, 추측/과장 금지. 링크는 그대로 유지한다.

[출력 형식 - 반드시 지킬 것]
- 제목:
- 분류: (뉴스/논문)
- 핵심 키워드: (중복 없는 단어 3개)
- 핵심 포인트:
  - (1)
  - (2)
  - (3)
- 기술 스택 태그: (예: Java/Spring | Python | TS/Node | MLOps 등)
- 개발자 관점 한 줄 평: (1문장)
- 참고 링크:

[항목]
{item_text}
""".strip()

        print(f"🧩 1차 요약 진행: {idx}/{len(items)}")
        resp = generate_with_retry(model_name, summary_prompt, retries=3)

        summaries.append(
            {
                "idx": idx,
                "type": it["type"],
                "title": it["title"],
                "link": it["link"],
                "summary_text": resp.text.strip(),
            }
        )

    return summaries


# -----------------------------
# 4) Reduce (고정 포맷 리포트)
# -----------------------------
def build_report_prompt(target_date: str, generated_time_kst: str, items, all_summaries_text: str):
    news_index_text, paper_index_text = build_raw_index(items)
    news_n = sum(1 for it in items if it.get("type") == "news")
    paper_n = sum(1 for it in items if it.get("type") == "paper")

    template_filled = REPORT_TEMPLATE.format(
        target_date=target_date,
        generated_time_kst=generated_time_kst,
        news_n=news_n,
        paper_n=paper_n,
    )

    report_prompt = f"""
너는 IT 전문 뉴스 큐레이터이자 시니어 개발자다.
아래 '구조화 요약'들과 '원문 목록'을 근거로, **반드시** 지정 템플릿 그대로 마크다운 리포트를 작성하라.

[출력 규칙 - 위반 금지]
1) 아래 템플릿의 섹션/헤더 이름을 **그대로** 사용한다. (추가 헤더 금지)
2) '오늘의 Top 이슈'는 3~5개.
3) 각 이슈 섹션은 반드시 다음 하위 구조를 포함한다:
   - 요약(불릿 3~5개)
   - 개발자 관점 한 줄 평(1문장)
   - 지금 바로 적용 아이디어(1~3개)
   - 리스크/주의(1~2개)
   - 참고 링크(최소 2개, 원문 목록에서 선택)
4) 마지막에 반드시 '오늘의 실무 액션 3가지'와 '원문 목록 (Raw Index)'를 포함한다.
5) 한국어로만 작성, 과장/추측 금지. 링크는 원문 그대로 복사해서 사용한다.

[반드시 이 템플릿 형식 그대로 출력]
{template_filled}

[구조화 요약들]
{all_summaries_text}

[원문 목록 - 반드시 링크는 여기서 사용]
## 원문 목록 (Raw Index)
### 뉴스
{news_index_text if news_index_text else "- (수집된 뉴스 링크 없음)"}

### 논문
{paper_index_text if paper_index_text else "- (수집된 논문 링크 없음)"}
""".strip()

    return report_prompt


def reduce_report(model_name: str, report_prompt: str):
    print("📰 2차 리포트 생성 시작")
    resp = generate_with_retry(model_name, report_prompt, retries=3)
    return resp.text.strip()


# -----------------------------
# 5) main
# -----------------------------
def main():
    # KST 기준 “어제” 날짜 리포트 생성
    kst = timezone(timedelta(hours=9))
    target_date = (datetime.now(kst) - timedelta(days=1)).strftime("%Y-%m-%d")
    generated_time_kst = datetime.now(kst).strftime("%H:%M")

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 GOOGLE_API_KEY가 필요합니다. GitHub Secrets로 넣으세요.")

    genai.configure(api_key=api_key)
    model_name = choose_model()

    print(f"🚀 AI Daily Report 시작 | target_date={target_date} | model={model_name}")

    items = collect_items(target_date, news_n=5, paper_n=5)
    print(f"✅ 수집 완료: {len(items)}건")

    summaries = map_summaries(model_name, items)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # 요약 저장
    summary_path = reports_dir / f"{target_date}_summaries.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 요약 저장: {summary_path}")

    # 리포트 생성
    all_summaries_text = "\n\n".join([s["summary_text"] for s in summaries])
    report_prompt = build_report_prompt(target_date, generated_time_kst, items, all_summaries_text)

    report_text = reduce_report(model_name, report_prompt)

    # 포맷 검사 + 끊김이면 1회 이어쓰기
    missing = validate_report_format(report_text)
    if missing or looks_truncated(report_text):
        print(f"⚠️ 포맷 누락/끊김 의심: {missing if missing else '(누락 없음, 끊김 의심)'}")
        print("➡️ 1회 이어쓰기 시도")
        report_text = continue_report(model_name, report_text)

    missing2 = validate_report_format(report_text)
    if missing2:
        print(f"❌ 최종 포맷 누락: {missing2}")
    else:
        print("✅ 포맷 검증 통과")

    # 리포트 저장
    report_path = reports_dir / f"{target_date}_AI_Report.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"💾 리포트 저장: {report_path}")

    print("🎉 종료")


if __name__ == "__main__":
    main()
