import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

from ddgs import DDGS
import arxiv

# 기존의 google.generativeai 패키지가 아니라
# 신규 Google Gen AI SDK인 google-genai 패키지를 사용합니다.
from google import genai
from google.genai import types


# ---------------------------------------------------------
# 0) 기본 설정
# ---------------------------------------------------------
# GitHub Actions에서 GEMINI_MODEL 환경변수를 지정하지 않으면
# 기본적으로 gemini-2.5-flash 모델을 사용합니다.
DEFAULT_MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------
# 1) 고정 포맷 템플릿
# ---------------------------------------------------------
REPORT_TEMPLATE = """# [AI Daily] {target_date} 기술 동향

> 생성 시간(KST): {generated_time_kst}
> 데이터 소스: DDGS(뉴스 {news_n}), arXiv(cs.AI/cs.LG, 논문 {paper_n})

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


# ---------------------------------------------------------
# 2) Gemini 공용 함수
# ---------------------------------------------------------
def choose_model() -> str:
    """
    사용할 Gemini 모델을 결정합니다.

    GitHub Actions 또는 로컬 환경에서 GEMINI_MODEL을 지정하면
    해당 값을 우선 사용합니다.

    환경변수가 없으면 DEFAULT_MODEL을 사용합니다.

    예:
        GEMINI_MODEL=gemini-2.5-flash
    """
    model_name = os.environ.get(
        "GEMINI_MODEL",
        DEFAULT_MODEL,
    ).strip()

    # 모델 이름을 다음처럼 입력한 경우:
    # models/gemini-2.5-flash
    #
    # 신규 SDK에 전달하기 쉬운 다음 형태로 변환합니다:
    # gemini-2.5-flash
    return model_name.removeprefix("models/")


def get_response_text(response) -> str:
    """
    Gemini 응답 객체에서 텍스트를 안전하게 추출합니다.

    안전 필터, 모델 오류, 비정상 응답 등의 이유로
    response.text가 비어 있을 수 있습니다.

    빈 응답을 그대로 파일로 저장하지 않도록 예외를 발생시킵니다.
    """
    text = getattr(response, "text", None)

    if not text or not text.strip():
        raise RuntimeError(
            "Gemini가 비어 있는 응답을 반환했습니다."
        )

    return text.strip()


def generate_with_retry(
    client,
    model_name: str,
    prompt_text: str,
    retries: int = 3,
    max_output_tokens: int = 4096,
    temperature: float = 0.2,
) -> str:
    """
    Gemini에 텍스트 생성을 요청합니다.

    다음과 같은 일시적 오류는 재시도합니다.

    - HTTP 429
    - API 호출량 제한
    - API 쿼터 제한
    - HTTP 500, 502, 503, 504
    - 네트워크 타임아웃
    - 일시적 서비스 장애

    재시도 간격:
    - 첫 번째 실패 후 30초
    - 두 번째 실패 후 60초
    """
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )

            return get_response_text(response)

        except Exception as error:
            last_error = error

            # google-genai의 API 오류 객체에는
            # code 속성이 들어 있는 경우가 있습니다.
            error_code = getattr(error, "code", None)
            error_message = str(error).lower()

            retryable = (
                error_code in {429, 500, 502, 503, 504}
                or "429" in error_message
                or "quota" in error_message
                or "rate limit" in error_message
                or "resource exhausted" in error_message
                or "timeout" in error_message
                or "temporarily unavailable" in error_message
                or "service unavailable" in error_message
            )

            if retryable and attempt < retries:
                wait_time = attempt * 30

                print(
                    f"⚠️ Gemini 일시 오류: "
                    f"{wait_time}초 후 재시도 "
                    f"({attempt}/{retries}) | {error}"
                )

                time.sleep(wait_time)
                continue

            # 재시도 대상이 아니거나
            # 최종 재시도까지 실패한 경우 예외를 다시 발생시킵니다.
            raise

    raise RuntimeError(
        "Gemini 호출 재시도 횟수를 초과했습니다."
    ) from last_error


# ---------------------------------------------------------
# 3) 리포트 공용 함수
# ---------------------------------------------------------
def build_raw_index(items):
    """
    Raw Index는 Gemini가 아니라 코드가 직접 생성합니다.

    목적:
    - 원문 링크 누락 방지
    - Gemini가 URL을 임의로 변경하는 문제 방지
    - 실제 수집한 링크만 보고서에 포함
    """
    news_lines = []
    paper_lines = []

    for item in items:
        title = (item.get("title") or "").strip()
        link = (item.get("link") or "").strip()

        if not title or not link:
            continue

        if item.get("type") == "news":
            news_lines.append(
                f"- {title} — {link}"
            )

        elif item.get("type") == "paper":
            paper_lines.append(
                f"- {title} — {link}"
            )

    return (
        "\n".join(news_lines),
        "\n".join(paper_lines),
    )


def validate_report_format(report_text: str):
    """
    최종 리포트가 필수 헤더를 포함하는지 검사합니다.

    반환값이 빈 리스트라면
    모든 필수 헤더가 정상적으로 포함된 것입니다.
    """
    required_headers = [
        "# [AI Daily]",
        "## 오늘의 Top 이슈",
        "## 오늘의 실무 액션 3가지",
        "## 원문 목록 (Raw Index)",
        "### 뉴스",
        "### 논문",
    ]

    return [
        header
        for header in required_headers
        if header not in report_text
    ]


def looks_truncated(report_text: str) -> bool:
    """
    Gemini 출력이 중간에 끊겼는지 간단하게 검사합니다.

    다음 조건 중 하나라도 충족하면
    출력이 중단된 것으로 판단합니다.

    1. 전체 결과가 1,200자 미만
    2. 마지막 Raw Index 섹션이 없음
    """
    if len(report_text) < 1200:
        return True

    if "## 원문 목록 (Raw Index)" not in report_text:
        return True

    return False


def continue_report(
    client,
    model_name: str,
    existing_text: str,
) -> str:
    """
    최종 리포트가 중간에 끊긴 경우
    마지막 부분부터 한 번 이어서 작성합니다.

    기존 리포트 전체가 아니라 마지막 1,500자만 전달하여
    중복 생성과 추가 토큰 사용을 줄입니다.
    """
    tail = existing_text[-1500:]

    prompt = f"""
아래 글은 작성 도중 중단된 AI 기술 동향 리포트다.

기존 내용을 반복하거나 처음부터 다시 작성하지 말고,
끊긴 지점의 다음 내용만 이어서 작성하라.

마크다운 형식을 유지하고 반드시 다음 섹션까지 완성하라.

- ## 오늘의 실무 액션 3가지
- ## 원문 목록 (Raw Index)
- ### 뉴스
- ### 논문

--- 기존 글의 마지막 부분 ---
{tail}

--- 이어서 작성 ---
""".strip()

    continuation = generate_with_retry(
        client=client,
        model_name=model_name,
        prompt_text=prompt,
        retries=3,
        max_output_tokens=4096,
        temperature=0.1,
    )

    return (
        existing_text.rstrip()
        + "\n\n"
        + continuation
    )


def deduplicate_items(items):
    """
    동일한 URL이 여러 번 검색됐을 때
    첫 번째 항목만 남깁니다.

    URL이 없다면 제목을 보조 중복 기준으로 사용합니다.
    """
    unique_items = []
    seen_keys = set()

    for item in items:
        link = (
            item.get("link")
            or ""
        ).strip().lower()

        title = (
            item.get("title")
            or ""
        ).strip().lower()

        key = link or title

        if not key:
            continue

        if key in seen_keys:
            continue

        seen_keys.add(key)
        unique_items.append(item)

    return unique_items


# ---------------------------------------------------------
# 4) 뉴스 수집
# ---------------------------------------------------------
def collect_news(
    target_date: str,
    news_n: int = 5,
):
    """
    DDGS를 이용하여 AI 관련 뉴스를 수집합니다.

    1차:
        DDGS.news()를 사용합니다.

    2차:
        뉴스 검색이 실패하거나 결과가 없는 경우
        DDGS.text()를 사용합니다.

    검색 서비스가 일시적으로 실패하더라도
    arXiv 논문 수집은 계속 진행할 수 있도록
    뉴스 검색 오류는 이 함수 내부에서 처리합니다.
    """
    if news_n <= 0:
        return []

    query = (
        f"artificial intelligence "
        f"technology news {target_date}"
    )

    ddgs = DDGS(timeout=20)
    results = []

    # 뉴스 전용 검색을 먼저 시도합니다.
    try:
        results = ddgs.news(
            query=query,
            region="us-en",
            safesearch="moderate",

            # 최근 하루 범위의 뉴스를 우선 검색합니다.
            timelimit="d",

            max_results=news_n,
        ) or []

    except Exception as error:
        print(
            f"⚠️ DDGS 뉴스 검색 실패: {error}"
        )
        print(
            "➡️ DDGS 일반 텍스트 검색으로 대체합니다."
        )

    # 뉴스 검색 결과가 없으면
    # 일반 웹 검색을 사용합니다.
    if not results:
        try:
            results = ddgs.text(
                query=query,
                region="us-en",
                safesearch="moderate",

                # 일반 검색은 최근 일주일 범위로 넓힙니다.
                timelimit="w",

                max_results=news_n,
            ) or []

        except Exception as error:
            print(
                f"⚠️ DDGS 텍스트 검색 실패: {error}"
            )
            return []

    news_items = []

    for result in results:
        title = (
            result.get("title")
            or ""
        ).strip()

        body = (
            result.get("body")
            or result.get("excerpt")
            or result.get("description")
            or ""
        ).strip()

        # DDGS.news()는 주로 url을 반환하고,
        # DDGS.text()는 주로 href를 반환합니다.
        #
        # 두 검색 방식에 모두 대응하기 위해
        # url과 href를 순서대로 확인합니다.
        link = (
            result.get("url")
            or result.get("href")
            or ""
        ).strip()

        # 제목이나 링크가 없는 결과는 제외합니다.
        if not title or not link:
            continue

        news_items.append(
            {
                "type": "news",
                "title": title,
                "body": body,
                "link": link,
            }
        )

        if len(news_items) >= news_n:
            break

    return news_items


# ---------------------------------------------------------
# 5) arXiv 논문 수집
# ---------------------------------------------------------
def collect_papers(paper_n: int = 5):
    """
    arXiv에서 최신 AI·머신러닝 논문을 수집합니다.

    기존 코드의 오류:

        search.results()

    최신 arxiv 패키지에서는 Search 객체에
    results() 메서드가 없습니다.

    최신 사용 방식:

        client = arxiv.Client()
        client.results(search)
    """
    if paper_n <= 0:
        return []

    arxiv_client = arxiv.Client(
        # 한 페이지에서 가져올 결과 수입니다.
        page_size=max(paper_n, 10),

        # arXiv API에 지나치게 빠르게 요청하지 않도록
        # 요청 사이에 3초 간격을 둡니다.
        delay_seconds=3.0,

        # 일시적인 통신 오류가 발생하면
        # 최대 3회까지 재시도합니다.
        num_retries=3,
    )

    search = arxiv.Search(
        # 인공지능 또는 머신러닝 카테고리를 검색합니다.
        query="(cat:cs.AI OR cat:cs.LG)",

        max_results=paper_n,

        # 논문 제출일을 기준으로 정렬합니다.
        sort_by=arxiv.SortCriterion.SubmittedDate,

        # 가장 최근 논문부터 가져옵니다.
        sort_order=arxiv.SortOrder.Descending,
    )

    paper_items = []

    try:
        # 핵심 수정 부분입니다.
        #
        # 기존:
        # for result in search.results():
        #
        # 변경:
        # for result in arxiv_client.results(search):
        for result in arxiv_client.results(search):
            title = (
                result.title
                or ""
            ).strip()

            body = (
                result.summary
                or ""
            ).strip()

            link = (
                result.pdf_url
                or ""
            ).strip()

            if not title or not link:
                continue

            paper_items.append(
                {
                    "type": "paper",
                    "title": title,
                    "body": body,
                    "link": link,
                }
            )

            if len(paper_items) >= paper_n:
                break

    except Exception as error:
        print(
            f"⚠️ arXiv 논문 수집 실패: {error}"
        )

    return paper_items


def collect_items(
    target_date: str,
    news_n: int = 5,
    paper_n: int = 5,
):
    """
    뉴스와 논문을 함께 수집하고
    중복 항목을 제거합니다.

    뉴스와 arXiv가 모두 실패해서
    자료가 한 건도 없는 경우에는
    근거 없는 보고서를 만들지 않고 작업을 중단합니다.
    """
    items = []

    items.extend(
        collect_news(
            target_date=target_date,
            news_n=news_n,
        )
    )

    items.extend(
        collect_papers(
            paper_n=paper_n,
        )
    )

    items = deduplicate_items(items)

    if not items:
        raise RuntimeError(
            "뉴스와 arXiv 논문을 "
            "한 건도 수집하지 못했습니다."
        )

    return items


# ---------------------------------------------------------
# 6) Map: 개별 자료 구조화 요약
# ---------------------------------------------------------
def map_summaries(
    client,
    model_name: str,
    items,
):
    """
    수집한 뉴스와 논문을 한 건씩
    Gemini에 전달하여 구조화된 요약을 생성합니다.
    """
    summaries = []

    for index, item in enumerate(
        items,
        start=1,
    ):
        item_text = f"""
[타입] {item['type']}
[제목] {item['title']}
[본문]
{item['body']}
[링크] {item['link']}
""".strip()

        summary_prompt = f"""
너는 시니어 개발자 관점의 AI 뉴스·논문 분석가다.

아래 자료만 근거로 한국어 구조화 요약을 작성하라.

자료에 없는 내용을 추측하거나 과장하지 말고,
원문 링크는 수정하지 말고 그대로 유지하라.

[출력 형식]
- 제목:
- 분류: 뉴스 또는 논문
- 핵심 키워드: 중복 없는 단어 3개
- 핵심 포인트:
  - 첫 번째 핵심 내용
  - 두 번째 핵심 내용
  - 세 번째 핵심 내용
- 기술 스택 태그:
- 개발자 관점 한 줄 평:
- 참고 링크:

[분석할 자료]
{item_text}
""".strip()

        print(
            f"🧩 1차 요약 진행: "
            f"{index}/{len(items)}"
        )

        summary_text = generate_with_retry(
            client=client,
            model_name=model_name,
            prompt_text=summary_prompt,
            retries=3,

            # 개별 항목 하나의 요약이므로
            # 지나치게 긴 출력을 방지합니다.
            max_output_tokens=1600,

            # 사실 중심의 비교적 일정한 결과를 위해
            # 낮은 temperature를 사용합니다.
            temperature=0.1,
        )

        summaries.append(
            {
                "idx": index,
                "type": item["type"],
                "title": item["title"],
                "link": item["link"],
                "summary_text": summary_text,
            }
        )

    return summaries


# ---------------------------------------------------------
# 7) Reduce: 최종 리포트 생성
# ---------------------------------------------------------
def build_report_prompt(
    target_date: str,
    generated_time_kst: str,
    items,
    all_summaries_text: str,
):
    """
    개별 요약과 실제 원문 URL을 결합하여
    최종 리포트 생성 프롬프트를 만듭니다.
    """
    (
        news_index_text,
        paper_index_text,
    ) = build_raw_index(items)

    news_count = sum(
        1
        for item in items
        if item.get("type") == "news"
    )

    paper_count = sum(
        1
        for item in items
        if item.get("type") == "paper"
    )

    template_filled = REPORT_TEMPLATE.format(
        target_date=target_date,
        generated_time_kst=generated_time_kst,
        news_n=news_count,
        paper_n=paper_count,
    )

    report_prompt = f"""
너는 IT 전문 뉴스 큐레이터이자 시니어 개발자다.

아래 구조화 요약과 원문 목록만 근거로
지정된 형식의 마크다운 리포트를 작성하라.

[출력 규칙]
1. 아래 템플릿의 섹션과 헤더 이름을 그대로 사용한다.

2. 오늘의 Top 이슈는
   수집된 자료 중 3~5개를 선정한다.

3. 동일한 자료를 여러 이슈에서 반복하지 않는다.

4. 각 이슈는 반드시 다음 구조를 포함한다.
   - 요약: 불릿 3~5개
   - 개발자 관점 한 줄 평: 1문장
   - 지금 바로 적용 아이디어: 1~3개
   - 리스크/주의: 1~2개
   - 참고 링크: 실제 원문 링크 1~2개

5. 마지막에 반드시 다음 섹션을 포함한다.
   - 오늘의 실무 액션 3가지
   - 원문 목록 (Raw Index)

6. 한국어로만 작성한다.

7. 자료에 없는 사실을 추측하거나 과장하지 않는다.

8. URL을 수정하거나 새로 만들지 않는다.

9. 뉴스 또는 논문이 없는 경우
   해당 Raw Index에
   '(수집된 항목 없음)'이라고 표시한다.

[반드시 이 템플릿 형식으로 출력]
{template_filled}

[구조화 요약]
{all_summaries_text}

[코드에서 생성한 원문 목록]
## 원문 목록 (Raw Index)

### 뉴스
{news_index_text if news_index_text else "- (수집된 뉴스 없음)"}

### 논문
{paper_index_text if paper_index_text else "- (수집된 논문 없음)"}
""".strip()

    return report_prompt


def reduce_report(
    client,
    model_name: str,
    report_prompt: str,
) -> str:
    """
    구조화된 개별 요약을 기반으로
    최종 AI 기술 동향 리포트를 생성합니다.
    """
    print("📰 2차 리포트 생성 시작")

    return generate_with_retry(
        client=client,
        model_name=model_name,
        prompt_text=report_prompt,
        retries=3,

        # 최종 보고서는 길어질 수 있으므로
        # 개별 요약보다 큰 출력 제한을 사용합니다.
        max_output_tokens=8192,

        temperature=0.2,
    )


# ---------------------------------------------------------
# 8) main
# ---------------------------------------------------------
def main():
    """
    KST 기준 전날의 AI 기술 동향 보고서를 생성합니다.

    생성 파일:
    - reports/YYYY-MM-DD_AI_Report.md
    - reports/YYYY-MM-DD_summaries.json
    """
    kst = timezone(
        timedelta(hours=9)
    )

    now_kst = datetime.now(kst)

    # KST 기준 전날 날짜를 보고서 날짜로 사용합니다.
    target_date = (
        now_kst - timedelta(days=1)
    ).strftime("%Y-%m-%d")

    generated_time_kst = now_kst.strftime(
        "%H:%M"
    )

    # 보고서 저장 폴더를 생성합니다.
    reports_dir = Path("reports")

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        reports_dir
        / f"{target_date}_AI_Report.md"
    )

    summary_path = (
        reports_dir
        / f"{target_date}_summaries.json"
    )

    # 보고서와 요약 파일이 모두 존재하면
    # API 호출과 외부 검색을 수행하지 않고 종료합니다.
    if (
        report_path.exists()
        and summary_path.exists()
    ):
        print(
            f"⏭️ 이미 생성됨: "
            f"{report_path.name}, "
            f"{summary_path.name} → 종료"
        )
        return

    # 신규 google-genai SDK는
    # GOOGLE_API_KEY 또는 GEMINI_API_KEY를 사용할 수 있습니다.
    api_key = (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY 또는 GEMINI_API_KEY "
            "환경변수가 필요합니다. "
            "GitHub Repository Secrets에 "
            "API 키를 등록하세요."
        )

    model_name = choose_model()

    # 신규 Google Gen AI SDK 클라이언트를 생성합니다.
    gemini_client = genai.Client(
        api_key=api_key
    )

    try:
        print(
            "🚀 AI Daily Report 시작 | "
            f"target_date={target_date} | "
            f"model={model_name}"
        )

        # -------------------------------------------------
        # 1. 뉴스와 논문 수집
        # -------------------------------------------------
        items = collect_items(
            target_date=target_date,
            news_n=5,
            paper_n=5,
        )

        news_count = sum(
            1
            for item in items
            if item.get("type") == "news"
        )

        paper_count = sum(
            1
            for item in items
            if item.get("type") == "paper"
        )

        print(
            f"✅ 수집 완료: 총 {len(items)}건 "
            f"(뉴스 {news_count}, 논문 {paper_count})"
        )

        # -------------------------------------------------
        # 2. 개별 자료 구조화 요약
        # -------------------------------------------------
        summaries = map_summaries(
            client=gemini_client,
            model_name=model_name,
            items=items,
        )

        # 개별 요약 결과를 JSON으로 저장합니다.
        summary_path.write_text(
            json.dumps(
                summaries,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"💾 요약 저장: {summary_path}"
        )

        # -------------------------------------------------
        # 3. 최종 리포트 생성
        # -------------------------------------------------
        all_summaries_text = "\n\n".join(
            summary["summary_text"]
            for summary in summaries
        )

        report_prompt = build_report_prompt(
            target_date=target_date,
            generated_time_kst=generated_time_kst,
            items=items,
            all_summaries_text=all_summaries_text,
        )

        report_text = reduce_report(
            client=gemini_client,
            model_name=model_name,
            report_prompt=report_prompt,
        )

        # -------------------------------------------------
        # 4. 포맷 누락 또는 출력 중단 검사
        # -------------------------------------------------
        missing_headers = validate_report_format(
            report_text
        )

        if (
            missing_headers
            or looks_truncated(report_text)
        ):
            print(
                "⚠️ 포맷 누락 또는 출력 중단 의심: "
                f"{missing_headers if missing_headers else '(헤더 누락 없음)'}"
            )

            print(
                "➡️ 이어쓰기 1회 시도"
            )

            report_text = continue_report(
                client=gemini_client,
                model_name=model_name,
                existing_text=report_text,
            )

        final_missing_headers = (
            validate_report_format(report_text)
        )

        if final_missing_headers:
            print(
                "❌ 최종 포맷 누락: "
                f"{final_missing_headers}"
            )
        else:
            print(
                "✅ 포맷 검증 통과"
            )

        # -------------------------------------------------
        # 5. 최종 리포트 저장
        # -------------------------------------------------
        report_path.write_text(
            report_text.rstrip() + "\n",
            encoding="utf-8",
        )

        print(
            f"💾 리포트 저장: {report_path}"
        )

        print("🎉 종료")

    finally:
        # 신규 google-genai SDK의 HTTP 연결 자원을
        # 명시적으로 정리합니다.
        gemini_client.close()


if __name__ == "__main__":
    main()